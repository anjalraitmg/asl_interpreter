import warnings
import sys
import os

# 1. Suppress the Protobuf warning immediately
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# 2. Robust Imports for CachyOS/Linux
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    import threading
    import queue
    from collections import deque
    import time
except (AttributeError, ImportError):
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# ---------------- CONFIG ----------------
WINDOW_NAME = "Pro Sign Interpreter"
BUFFER_SIZE = 10 
MIN_CONFIDENCE = 0.50 
SMOOTHING_WINDOW = 5

# Forced HD Resolution
CAM_WIDTH = 1280
CAM_HEIGHT = 720

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

# ---------------- MATH & HELPERS ----------------
def get_dist(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

def get_finger_ext(hand):
    if not hand: return [0.0]*5
    wrist = hand.landmark[0]
    res = []
    # 4 Fingers
    for tip, mcp in [(8,5), (12,9), (16,13), (20,17)]:
        res.append(get_dist(hand.landmark[tip], wrist) / (get_dist(hand.landmark[mcp], wrist) + 0.001))
    # Thumb
    res.insert(0, get_dist(hand.landmark[4], hand.landmark[17]) * 5) 
    return res

# ---------------- SIGN ENGINE ----------------
def evaluate_signs(results, history):
    if not results or not results.right_hand_landmarks or not results.pose_landmarks:
        history.clear()
        return None, 0.0

    rh = results.right_hand_landmarks
    lh = results.left_hand_landmarks
    pose = results.pose_landmarks
    
    wrist = rh.landmark[0]
    shoulder_r = pose.landmark[12]
    shoulder_l = pose.landmark[11]
    nose = pose.landmark[0]

    # Ruler: Shoulder width (distance-independent math)
    uom = get_dist(shoulder_r, shoulder_l)
    if uom < 0.05: return None, 0.0

    # Smooth Movement
    history.append((wrist.x, wrist.y))
    dx = history[-1][0] - history[0][0]
    dy = history[-1][1] - history[0][1]

    ext = get_finger_ext(rh) 
    is_fist = sum(1 for e in ext[1:] if e < 1.1) >= 3
    is_flat = sum(1 for e in ext[1:] if e > 1.25) >= 3
    pinch_dist = get_dist(rh.landmark[8], rh.landmark[4])

    scores = {}

    # 1. I LOVE YOU (Fuzzy logic for high accuracy)
    ily_pts = 0
    if ext[0] > 1.0: ily_pts += 0.25 # Thumb
    if ext[1] > 1.2: ily_pts += 0.25 # Index
    if ext[4] > 1.2: ily_pts += 0.25 # Pinky
    if ext[2] < 1.1 and ext[3] < 1.1: ily_pts += 0.25 # Middle/Ring down
    if ily_pts >= 0.75: scores["I LOVE YOU"] = ily_pts

    # 2. HELLO (Temple)
    if is_flat and wrist.y < shoulder_r.y and get_dist(wrist, nose) < uom * 1.2:
        scores["HELLO"] = 0.6 + abs(dx)*4

    # 3. PLEASE (Flat hand on chest)
    if is_flat and wrist.y > shoulder_r.y and shoulder_r.x < wrist.x < shoulder_l.x:
        scores["PLEASE"] = 0.7 + min(abs(dx)*5, 0.3)

    # 4. SORRY (Fist on chest)
    if is_fist and wrist.y > shoulder_r.y and shoulder_r.x < wrist.x < shoulder_l.x:
        scores["SORRY"] = 0.7 + min(abs(dx)*5, 0.3)

    # 5. HELP (R-Fist on L-Palm)
    if lh:
        l_ext = get_finger_ext(lh)
        l_flat = sum(1 for e in l_ext[1:] if e > 1.2) >= 3
        if is_fist and l_flat and get_dist(wrist, lh.landmark[0]) < uom * 0.4:
            scores["HELP"] = 0.9

    # 6. FRIEND (Touching Index Fingers)
    if lh:
        l_ext = get_finger_ext(lh)
        if ext[1] > 1.2 and l_ext[1] > 1.2 and get_dist(rh.landmark[8], lh.landmark[8]) < uom * 0.2:
            scores["FRIEND"] = 0.9

    # 7. EAT (Pinch near mouth)
    if pinch_dist < uom * 0.15 and get_dist(rh.landmark[4], nose) < uom * 0.5:
        scores["EAT"] = 0.9

    # 8. YES (Fist nodding)
    if is_fist and abs(dy) > 0.04:
        scores["YES"] = 0.8

    # 9. NO (Snap/Pinch)
    if pinch_dist < uom * 0.15 and not is_flat and abs(dy) > 0.02:
        scores["NO"] = 0.8

    # 10. GOODBYE (High Wave)
    if is_flat and wrist.y < shoulder_r.y - 0.1 and abs(dx) > 0.08:
        scores["GOODBYE"] = 0.8

    if not scores: return None, 0.0
    best = max(scores, key=scores.get)
    return best, scores[best]

# ---------------- WORKERS ----------------
def camera_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        if frame_queue.full(): frame_queue.get_nowait()
        frame_queue.put(frame)
    cap.release()

def processing_worker():
    history = deque(maxlen=BUFFER_SIZE)
    conf_buffer = deque(maxlen=SMOOTHING_WINDOW)
    with mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sign, conf = evaluate_signs(results, history)
                
                conf_buffer.append((sign, conf))
                valid = [c for c in conf_buffer if c[0]]
                final_sign, avg_conf = None, 0.0
                if valid:
                    final_sign = max(set(v[0] for v in valid), key=lambda s: [v[0] for v in valid].count(s))
                    avg_conf = sum(v[1] for v in valid if v[0] == final_sign) / len(valid)

                if result_queue.full(): result_queue.get_nowait()
                result_queue.put((frame, results, final_sign, avg_conf))
            except queue.Empty: continue
            except Exception: continue

# ---------------- MAIN ----------------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# Setting the window to high-res start
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

cam_t = threading.Thread(target=camera_worker, daemon=True)
proc_t = threading.Thread(target=processing_worker, daemon=True)
cam_t.start(); proc_t.start()

last_word, last_conf, expiry = "", 0.0, 0

try:
    while not stop_event.is_set():
        try:
            frame, results, sign, conf = result_queue.get(timeout=0.1)
            h, w = frame.shape[:2]

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if sign and conf > MIN_CONFIDENCE:
                last_word, last_conf, expiry = sign, conf, time.time() + 1.2

            if time.time() < expiry:
                # Dynamic UI scaling based on HD resolution
                bar_max_w, bar_h = int(w * 0.35), int(h * 0.08)
                bx, by = int(w * 0.03), int(h * 0.03)
                cv2.rectangle(frame, (bx, by), (bx + bar_max_w, by + bar_h), (30, 30, 30), -1)
                fill_w = int(bar_max_w * last_conf)
                cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 0), -1)
                cv2.putText(frame, f"{last_word} {int(last_conf * 100)}%", (bx + 15, by + int(bar_h * 0.7)),
                            cv2.FONT_HERSHEY_TRIPLEX, w/1100.0, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
        except queue.Empty: pass
        if cv2.waitKey(1) & 0xFF == 27: stop_event.set()
finally:
    stop_event.set()
    cam_t.join(timeout=1)
    proc_t.join(timeout=1)
    cv2.destroyAllWindows()
    sys.exit(0)