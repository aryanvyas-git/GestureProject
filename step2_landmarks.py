"""
STEP 2 — Webcam + Hand Landmark Detection
==========================================
Run this file to test that your webcam works and MediaPipe
can detect your hand landmarks.

You should see:
- A webcam window with your hand detected
- Green dots on 21 landmark points
- Connections drawn between them
- Press Q to quit
"""

import cv2
import mediapipe as mp

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands detector config:
# max_num_hands=1 (we only track one hand for now)
# min_detection_confidence=0.7 (how confident before declaring a hand found)
# min_tracking_confidence=0.5 (how confident to keep tracking between frames)
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Open webcam ---
cap = cv2.VideoCapture(0)  # 0 = default webcam. Try 1 if this doesn't work.

if not cap.isOpened():
    print("ERROR: Could not open webcam. Check that it's connected.")
    exit()

print("Webcam opened. Show your hand to the camera. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally (mirror effect — more natural for the user)
    frame = cv2.flip(frame, 1)

    # MediaPipe works with RGB, OpenCV uses BGR — convert
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    # --- Draw landmarks if a hand is detected ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the 21 dots and connecting lines
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Print the landmark coordinates to console (for learning)
            # Landmark 8 = tip of index finger
            index_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x_px = int(index_tip.x * w)
            y_px = int(index_tip.y * h)

            cv2.putText(
                frame,
                f"Index tip: ({x_px}, {y_px})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(frame, "Hand detected!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

    cv2.imshow("Step 2 - Hand Landmark Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")


# =====================================================================
# UNDERSTANDING THE 21 LANDMARKS
# =====================================================================
#
#  MediaPipe labels each landmark with an ID (0-20):
#
#       8   12  16  20       ← Fingertips
#       |   |   |   |
#       7   11  15  19
#       |   |   |   |
#       6   10  14  18
#       |   |   |   |
#       5   9   13  17
#        \  |   |  /
#    4    \ |   | /
#    |     \|   |/
#    3      0---0            ← 0 = Wrist
#    |      Wrist
#    2
#    |
#    1
#
#  Finger tip IDs: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
#  Finger MCP (base knuckle): Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
#
#  Each landmark has:
#    .x  (0.0 to 1.0, left to right in frame)
#    .y  (0.0 to 1.0, top to bottom in frame)
#    .z  (depth — smaller = closer to camera)
# =====================================================================
