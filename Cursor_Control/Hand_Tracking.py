'''This code is for mouse pointer control using camera.

    Pinch Index finger and Thumb: Left click
    Close Fist: Stop
    Open Fist: Move
    
    PINCH_THRESHOLD = 0.05 =>  Lower it → You must pinch tighter to trigger a drag.
                               Raise it → It'll detect a pinch even with a small gap.
                               
    MOVE_THRESHOLD = 10 =>  Low value (e.g. 5) → more sensitive → even tiny tremors move the cursor.
                            Higher value (e.g. 20-30) → steadier → small hand jitters are ignored.
                            
    SMOOTHING_WINDOW = 5 =>  Lower (e.g. 3) → faster response, more jitter.
                             Higher (e.g. 10) → smoother but slower.
                             
    pag.moveTo(avg_x, avg_y, duration=0.1) =>  Lower duration → cursor moves faster.
                                               Higher duration → slower, smoother movement.
'''

# import libraries
import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np
from collections import deque
import math

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a Hands object
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Screen size for cursor control
screen_w, screen_h = pag.size()

# Settings for smoothing
SMOOTHING_WINDOW = 5
MOVE_THRESHOLD = 15

# Keep history of positions
pos_history_x = deque(maxlen=SMOOTHING_WINDOW)
pos_history_y = deque(maxlen=SMOOTHING_WINDOW)

# Click threshold
PINCH_THRESHOLD = 0.05   # adjust as needed

# Maintain drag state across calls
drag_state = {"is_dragging": False, "last_x": None, "last_y": None}

def control_cursor(x, y, is_pinching, is_fist):
    """
    Handles all pyautogui calls:
    - smooths motion
    - applies threshold
    - controls mouse down / up for drag
    - stops movement if fist detected
    """
    if is_fist:
        # Release drag if previously dragging
        if drag_state["is_dragging"]:
            pag.mouseUp()
            drag_state["is_dragging"] = False
        return

    # Add current position to smoothing history
    pos_history_x.append(x)
    pos_history_y.append(y)

    avg_x = int(sum(pos_history_x) / len(pos_history_x))
    avg_y = int(sum(pos_history_y) / len(pos_history_y))

    # Check movement threshold
    last_x = drag_state["last_x"]
    last_y = drag_state["last_y"]

    if last_x is None or last_y is None:
        drag_state["last_x"] = avg_x
        drag_state["last_y"] = avg_y

    delta_x = abs(avg_x - drag_state["last_x"])
    delta_y = abs(avg_y - drag_state["last_y"])

    if (delta_x > MOVE_THRESHOLD) or (delta_y > MOVE_THRESHOLD):
        pag.moveTo(avg_x, avg_y, duration=0.1)
        drag_state["last_x"] = avg_x
        drag_state["last_y"] = avg_y

    # Handle dragging
    if is_pinching and not drag_state["is_dragging"]:
        pag.mouseDown()
        drag_state["is_dragging"] = True
    elif not is_pinching and drag_state["is_dragging"]:
        pag.mouseUp()
        drag_state["is_dragging"] = False

# Webcam control using cv2
def frame_processing(video):
    cap = cv2.VideoCapture(video)
    
    if not cap.isOpened():
        print("Error: Webcam not detected")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing or failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Prepare black background
        black_frame = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark

                # Check folded fingers
                folded_fingers = 0
                for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
                    if landmarks[tip].y > landmarks[pip].y:
                        folded_fingers += 1

                is_fist = folded_fingers == 4

                # Get thumb tip and index tip
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]

                # Calculate distance between thumb tip and index tip
                dist = math.hypot(
                    (thumb_tip.x - index_tip.x),
                    (thumb_tip.y - index_tip.y)
                )

                # Check if pinching
                is_pinching = dist < PINCH_THRESHOLD

                # Display gesture status
                if is_fist:
                    text = "Fist (Stopped)"
                    color = (0, 255, 255)
                elif is_pinching:
                    text = "Dragging"
                    color = (0, 0, 255)
                else:
                    text = "Moving"
                    color = (0, 255, 0)

                cv2.putText(black_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, color, 3)

                # Map thumb tip to screen coordinates
                screen_x = int(thumb_tip.x * screen_w)
                screen_y = int(thumb_tip.y * screen_h)

                # Call the unified control function
                control_cursor(screen_x, screen_y, is_pinching, is_fist)

        else:
            # Hand disappears → release drag if needed
            if drag_state["is_dragging"]:
                pag.mouseUp()
                drag_state["is_dragging"] = False

        # ✅ Resize black frame to half the screen size (1:2 ratio)
        resized_frame = cv2.resize(black_frame, (screen_w // 2, screen_h // 2))
        cv2.imshow("Gesture Control for Cursor", resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    video_path = 0
    try:
        frame_processing(video_path)
    except Exception as e:
        print("Error:", e)
