"""
STEP 3 — Rule-Based Gesture Classifier
=======================================
This file adds gesture recognition on top of Step 2.

Logic: For each finger, we check if it's EXTENDED or CURLED
by comparing the fingertip's y-coordinate to the knuckle below it.
(Remember: smaller y = higher up on screen)

Gestures recognised:
  - Open palm     → "Hello"
  - Fist          → "Stop"
  - Thumbs up     → "Good"
  - Peace sign    → "Peace"
  - Point (index) → "Yes / Next"
  - Pinky up      → "Okay / Done"

Press Q to quit.
"""

import cv2
import mediapipe as mp
import numpy as np

# ─── MediaPipe Setup ───────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ─── Landmark Index Constants ──────────────────────────────────────
# Tip IDs for each finger
TIPS      = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
# One joint below the tip (PIP joint = Proximal Interphalangeal)
PIP       = [3, 6, 10, 14, 18]
# Base knuckle (MCP)
MCP       = [2, 5,  9, 13, 17]


def get_finger_states(landmarks):
    """
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    True  = finger is EXTENDED (up)
    False = finger is CURLED (down)
    """
    states = []

    # ── Thumb ──────────────────────────────────────────────────────
    # Special case: thumb extends sideways, not vertically.
    # We compare x-coordinates instead of y.
    # If the thumb tip is further LEFT than the IP joint → extended
    # (This assumes a right hand — works for most demo purposes)
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    # We use x: tip.x < ip.x means extended to the left (right hand)
    # Flip condition for left hand, but we'll keep it simple for now
    thumb_extended = thumb_tip.x < thumb_ip.x
    states.append(thumb_extended)

    # ── Other 4 fingers ────────────────────────────────────────────
    # A finger is extended if its tip's y is LESS than its PIP joint y
    # (because y increases downward on screen — tip above pip = extended)
    for tip_id, pip_id in zip(TIPS[1:], PIP[1:]):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        states.append(tip.y < pip.y)

    return states  # [thumb, index, middle, ring, pinky]


def classify_gesture(finger_states):
    """
    Maps a combination of finger states to a gesture name.
    finger_states = [thumb, index, middle, ring, pinky]  (True = extended)
    """
    t, i, m, r, p = finger_states

    # Open palm — all 5 extended
    if t and i and m and r and p:
        return "HEY! 🖐 ️"

    # Fist — all 5 curled
    if not i and not m and not r and not p:
        if not t:
            return "Stop ✊"

    # Thumbs up — only thumb extended
    if t and not i and not m and not r and not p:
        return "Good 👍"

    # Thumbs down — only thumb extended but pointing down
    # (We'll detect this via thumb tip y > wrist y — a simple check)
    # Handled separately in the main loop below

    # Peace / Victory — index + middle extended, others curled
    if not t and i and m and not r and not p:
        return "Peace ✌️"

    # Pointing — only index extended
    if not t and i and not m and not r and not p:
        return "Next 👆"

    # Pinky up — only pinky extended
    if not t and not i and not m and not r and p:
        return "Done 🤙"

    # Three fingers — index + middle + ring
    if not t and i and m and r and not p:
        return "Three 🤟"

    # Rock on — index + pinky extended
    if not t and i and not m and not r and p:
        return "Rock 🤘"

    # OK sign — thumb + index (approximate: thumb + index curled together)
    # Tricky to detect purely by extension, so we use thumb + rest extended
    if t and not i and m and r and p:
        return "OK 👌"

    return "..."


# ─── Main Loop ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# Smooth out flickering: only update label if same gesture
# appears for N consecutive frames
STABILITY_THRESHOLD = 8
gesture_buffer = []
stable_gesture = "..."

print("Gesture classifier running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            lm = hand_landmarks.landmark
            finger_states = get_finger_states(lm)
            current_gesture = classify_gesture(finger_states)

            # Debug: show finger states as 0/1
            state_str = " ".join(["T" if s else "_" for s in finger_states])
            cv2.putText(frame, f"Fingers: {state_str}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Gesture stability buffer ───────────────────────────────────
    gesture_buffer.append(current_gesture)
    if len(gesture_buffer) > STABILITY_THRESHOLD:
        gesture_buffer.pop(0)

    # Accept a gesture only when it's been consistent
    if gesture_buffer.count(current_gesture) >= STABILITY_THRESHOLD - 1:
        stable_gesture = current_gesture

    # ── Display on frame ──────────────────────────────────────────
    # Background box for text
    cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {stable_gesture}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 2)

    cv2.imshow("Step 3 - Gesture Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
