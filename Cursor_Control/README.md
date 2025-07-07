# Hand Cursor Controller

Control your computer’s cursor using your hand movements detected through your webcam!

This project uses **OpenCV (cv2)** and **MediaPipe** to detect your hand in real time and translate its position into cursor movements on your screen.

## Features

✅ Real-time hand tracking using your webcam  
✅ Maps hand coordinates to control your system cursor  
✅ Smooth tracking and movement filtering  
✅ Easy to run on most computers with a webcam  

## Requirements

- Python 3.7 or higher
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [pyautogui](https://pypi.org/project/PyAutoGUI/) (for cursor control)

Install all dependencies using:

```bash
pip install opencv-python mediapipe pyautogui
