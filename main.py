import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. Initialize MediaPipe Hands tracking tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)

# 2. Setup Windows Default Emoji/TrueType Fonts
# Segoe UI Emoji is the default Windows font that supports full-color emojis
try:
    font_path = "C:\\Windows\\Fonts\\seguiemj.ttf" # Windows Emoji Font
    font = ImageFont.truetype(font_path, 40)
except IOError:
    try:
        font_path = "C:\\Windows\\Fonts\\arial.ttf" # Fallback if missing
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        font = ImageFont.load_default()

# 3. Open the webcam
cap = cv2.VideoCapture(0)

print("Smart Gesture Detector with Emojis Active! Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the video for a natural look
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Standard Tip Landmark IDs for [Thumb, Index, Middle, Ring, Pinky]
    tipIds = [4, 8, 12, 16, 20]
    
    # Default message if no specific gesture is recognized
    gesture_text = "Scanning..."
    text_color = (255, 255, 255) # White (RGB for Pillow)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                fingers = []
                # Thumb extended check
                if abs(lmList[tipIds[0]][1] - lmList[tipIds[0] - 2][1]) > 20:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Remaining 4 Fingers logic
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- Match Custom Gestures ---
                # HELLO / GREETING
                if fingers == [1, 1, 1, 1, 1]:
                    gesture_text = "Hello! 👋"
                    text_color = (255, 255, 0) # Yellow (RGB)

                # PEACE SIGN
                elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                    gesture_text = "Peace! ✌️"
                    text_color = (255, 0, 255) # Purple (RGB)

                # ROCK ON
                elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
                    gesture_text = "Rock On! 🤘"
                    text_color = (255, 165, 0) # Orange (RGB)

                # THUMB ONLY (Thumbs Up or Thumbs Down)
                elif fingers == [1, 0, 0, 0, 0]:
                    if lmList[4][2] < lmList[2][2]:
                        gesture_text = "Thumbs Up! 👍"
                        text_color = (0, 255, 255) # Cyan (RGB)
                    else:
                        gesture_text = "Lose! 👎"
                        text_color = (255, 0, 0) # Red (RGB)

            # Draw the tracking mesh lines on each hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Pillow Text Rendering Layer ---
        # 1. Draw the basic black background box using OpenCV
        cv2.rectangle(frame, (20, 20), (550, 140), (0, 0, 0), cv2.FILLED) 

        # 2. Convert OpenCV BGR image to PIL RGB image to draw the text/emoji
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        
        # 3. Draw text using our emoji-compatible font
        draw.text((40, 50), gesture_text, font=font, fill=text_color)
        
        # 4. Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    # Show the frame
    cv2.imshow('Smart Gesture Console', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()