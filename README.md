# Real-Time Sign Language Interpreter 🤟

A high-performance, real-time computer vision application that translates basic sign language gestures into text on-screen. 

This project uses **OpenCV** for video rendering and **Google MediaPipe** for holistic body and hand tracking. To ensure smooth video playback without freezing, the architecture utilizes Python `threading` and `queue` systems to separate the camera capture loop from the heavy Machine Learning processing loop.

## Features
* **Real-Time Tracking:** Detects face, pose, and dual-hand landmarks.
* **Multithreaded Architecture:** Maintains high FPS by running inference in a background thread.
* **Dynamic Resolution:** Automatically scales UI elements based on camera resolution (defaults to 720p HD).
* **Modern Package Management:** Built and managed using `uv`.

## Supported Signs
Currently, the heuristic engine can detect the following signs based on finger extension, spatial distance, and movement history:
* I LOVE YOU, HELLO, PLEASE, SORRY, HELP, FRIEND, EAT, YES, NO, GOODBYE

## Installation and Usage

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/your-repo-name.git
   cd your-repo-name

2. Sync the dependencies and run the application:
   uv run main.py
   
   
