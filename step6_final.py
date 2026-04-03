"""
GESTURE RECOGNITION — Right Hand Only
======================================
All gestures are on the RIGHT HAND.

GESTURE → WORD MAPPING:
  Shake wrist          → Hi
  Fist                 → project
  Index + middle up    → is
  Index point at cam   → this
  Thumb sideways       → my
  Pinch (🤏)           → minor
  Open palm            → Thank You
  Thumbs up            → Good
  Index only up        → No
  Pinky up             → More
  Three fingers        → Want
  OK                   → Done

CONTROLS:
  SPACE     → Speak full sentence
  BACKSPACE → Remove last word
  C         → Clear sentence
  Q         → Quit
"""

import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import math


# ══════════════════════════════════════════════════════════════════════
#  LANDMARK CONSTANTS
# ══════════════════════════════════════════════════════════════════════

TIPS = [4, 8, 12, 16, 20]
PIP  = [3, 6, 10, 14, 18]


# ══════════════════════════════════════════════════════════════════════
#  FINGER STATE DETECTION
# ══════════════════════════════════════════════════════════════════════

def get_finger_states(landmarks):
    """
    Returns [thumb, index, middle, ring, pinky] as True/False.
    True = extended, False = curled. Right hand only.
    """
    states = []
    # Thumb: x-axis comparison for right hand
    states.append(landmarks[4].x < landmarks[3].x)
    # Other 4 fingers: tip above PIP joint = extended
    for tip_id, pip_id in zip(TIPS[1:], PIP[1:]):
        states.append(landmarks[tip_id].y < landmarks[pip_id].y)
    return states  # [thumb, index, middle, ring, pinky]


# ══════════════════════════════════════════════════════════════════════
#  SPECIAL GESTURE DETECTORS
# ══════════════════════════════════════════════════════════════════════

def landmark_distance(a, b):
    """Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def is_pinch(landmarks):
    """
    Pinch gesture 🤏 — thumb tip and index tip very close together,
    all other fingers curled.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    dist = landmark_distance(thumb_tip, index_tip)

    # Middle, ring, pinky must NOT be extended
    for tip_id, pip_id in zip(TIPS[2:], PIP[2:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False

    return dist < 0.06


def is_pointing_at_camera(landmarks):
    """
    Index finger pointing INTO the camera (toward you).
    Index extended AND tip z significantly closer than its base.
    All other fingers curled.
    """
    tip = landmarks[8]
    pip = landmarks[6]
    mcp = landmarks[5]

    # Index must be extended
    if not (tip.y < pip.y):
        return False

    # Middle, ring, pinky must be curled
    for tip_id, pip_id in zip(TIPS[2:], PIP[2:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False

    # Tip z much smaller (closer) than MCP z = pointing at camera
    return (mcp.z - tip.z) > 0.06


def is_thumb_sideways(landmarks):
    """
    Thumb pointing horizontally sideways.
    Only thumb extended AND it's roughly horizontal (not pointing up).
    """
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist     = landmarks[0]

    # All fingers must be curled
    for tip_id, pip_id in zip(TIPS[1:], PIP[1:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False

    # Thumb horizontal: tip and base at similar y level
    vertical_diff    = abs(thumb_tip.y - thumb_mcp.y)
    horizontal_reach = abs(thumb_tip.x - wrist.x)

    return vertical_diff < 0.08 and horizontal_reach > 0.10


# ══════════════════════════════════════════════════════════════════════
#  MAIN GESTURE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

def classify_gesture(finger_states, landmarks):
    """
    Classifies right hand gesture into a word.
    Special gestures are checked first before finger-state patterns.
    """
    t, i, m, r, p = finger_states

    # ── Special gestures (check FIRST — they override simple patterns)

    # Pinch 🤏 → "minor"
    if is_pinch(landmarks):
        return "Ajay don"

    # Thumb sideways → "my"
    if is_thumb_sideways(landmarks):
        return "my"

    # Index pointing at camera → "this"
    # (checked after pinch/thumb so no conflict)
    if not t and i and not m and not r and not p:
        if is_pointing_at_camera(landmarks):
            return "this"

    # ── Standard finger-state gestures ────────────────────────────

    # Open palm — all 5 extended → "Thank You"
    if t and i and m and r and p:
        return "Hi"

    # Fist — all curled → "project"
    if not t and not i and not m and not r and not p:
        return "project"

    # Thumbs up — only thumb → "Good"
    if t and not i and not m and not r and not p:
        return "Good"

    # Index + middle up (V sign) → "is"
    if not t and i and m and not r and not p:
        return "is"

    # Index only → "No"  (plain point, not at camera)
    if not t and i and not m and not r and not p:
        return "No"

    # Pinky only → "More"
    if not t and not i and not m and not r and p:
        return "More"

    # Three fingers (index + middle + ring) → "Want"
    if not t and i and m and r and not p:
        return "Want"

    # OK (thumb + middle + ring + pinky extended) → "Done"
    if t and not i and m and r and p:
        return "Done"

    return None


# ══════════════════════════════════════════════════════════════════════
#  SHAKE DETECTOR
# ══════════════════════════════════════════════════════════════════════

class ShakeDetector:
    def __init__(self, min_reversals=3, window=18, min_delta=0.03):
        self.min_reversals   = min_reversals
        self.window          = window
        self.min_delta       = min_delta
        self.positions       = []
        self.last_shake_time = 0
        self.cooldown        = 2.0

    def update(self, wrist_x):
        self.positions.append(wrist_x)
        if len(self.positions) > self.window:
            self.positions.pop(0)
        if len(self.positions) < self.window:
            return False

        reversals = 0
        prev_dir  = 0
        for i in range(1, len(self.positions)):
            delta = self.positions[i] - self.positions[i - 1]
            if abs(delta) < self.min_delta:
                continue
            curr_dir = 1 if delta > 0 else -1
            if prev_dir != 0 and curr_dir != prev_dir:
                reversals += 1
            prev_dir = curr_dir

        now = time.time()
        if reversals >= self.min_reversals and (now - self.last_shake_time) > self.cooldown:
            self.last_shake_time = now
            self.positions.clear()
            return True
        return False


# ══════════════════════════════════════════════════════════════════════
#  SPEECH ENGINE
# ══════════════════════════════════════════════════════════════════════

class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        self.lock        = threading.Lock()
        self.is_speaking = False
        self.last_spoken = {}

    def speak(self, text, force=False, cooldown=2.5):
        now = time.time()
        with self.lock:
            if not force:
                if text in self.last_spoken and now - self.last_spoken[text] < cooldown:
                    return
            if self.is_speaking:
                return
            self.last_spoken[text] = now
            self.is_speaking = True

        def _run():
            self.engine.say(text)
            self.engine.runAndWait()
            with self.lock:
                self.is_speaking = False

        threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
#  GESTURE STABILITY BUFFER
# ══════════════════════════════════════════════════════════════════════

class GestureBuffer:
    def __init__(self, size=12):
        self.size       = size
        self.buffer     = []
        self.stable     = None
        self.last_added = None

    def update(self, gesture):
        self.buffer.append(gesture)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

        if len(self.buffer) == self.size:
            counts = {}
            for g in self.buffer:
                if g:
                    counts[g] = counts.get(g, 0) + 1
            if counts:
                top, count = max(counts.items(), key=lambda x: x[1])
                self.stable = top if count >= self.size - 2 else None
            else:
                self.stable = None
        return self.stable

    def reset_last(self):
        self.last_added = None


# ══════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════

FONT     = cv2.FONT_HERSHEY_SIMPLEX
C_GRAY   = (160, 160, 160)
C_GREEN  = (80, 220, 100)
C_ORANGE = (60, 160, 255)
C_ACCENT = (220, 220, 80)


def draw_panel(frame, x1, y1, x2, y2, alpha=0.72):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, stable_gesture, sentence, is_speaking, fps):
    h, w = frame.shape[:2]

    # ── Top bar ────────────────────────────────────────────────────
    draw_panel(frame, 0, 0, w, 70)

    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 22), FONT, 0.5, C_GRAY, 1)

    status       = "◉ Speaking..." if is_speaking else "○ Ready"
    status_color = C_GREEN if is_speaking else C_GRAY
    cv2.putText(frame, status, (w - 130, 50), FONT, 0.45, status_color, 1)

    cv2.putText(frame, "Gesture:", (15, 28), FONT, 0.6, C_GRAY, 1)
    g_display = stable_gesture if stable_gesture else "—"
    cv2.putText(frame, g_display, (105, 28), FONT, 0.75, C_ORANGE, 2)

    # ── Sentence bar (bottom) ──────────────────────────────────────
    draw_panel(frame, 0, h - 85, w, h)

    sentence_str = " ".join(sentence) if sentence else "(make a gesture...)"
    if len(sentence_str) > 55:
        sentence_str = "..." + sentence_str[-52:]

    cv2.putText(frame, "Sentence:", (15, h - 58), FONT, 0.55, C_GRAY, 1)
    cv2.putText(frame, sentence_str, (15, h - 30), FONT, 0.85, C_ACCENT, 2)

    controls = "SPACE: Speak  |  BACKSPACE: Undo  |  C: Clear  |  Q: Quit"
    cv2.putText(frame, controls, (15, h - 10), FONT, 0.4, C_GRAY, 1)

    # ── Gesture guide ──────────────────────────────────────────────
    guide = [
        ("Shake wrist",     "Hi"),
        ("Fist",            "project"),
        ("Index+middle up", "is"),
        ("Point at camera", "this"),
        ("Thumb sideways",  "my"),
        ("Pinch",           "minor"),
        ("Open palm",       "Thank You"),
        ("Thumb up",        "Good"),
        ("Index only",      "No"),
        ("Pinky up",        "More"),
    ]
    gx = w - 210
    gy = 80
    draw_panel(frame, gx, gy, gx + 200, gy + len(guide) * 26 + 30)
    cv2.putText(frame, "RIGHT HAND", (gx + 8, gy + 18), FONT, 0.48, C_ORANGE, 1)
    for idx, (name, word) in enumerate(guide):
        y = gy + 36 + idx * 26
        cv2.putText(frame, name,        (gx + 8,   y), FONT, 0.4, C_GRAY,   1)
        cv2.putText(frame, f"→ {word}", (gx + 125, y), FONT, 0.4, C_ORANGE, 1)


# ══════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    mp_hands          = mp.solutions.hands
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands_detector = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.6
    )

    speech    = SpeechEngine()
    shake_det = ShakeDetector(min_reversals=3, window=18, min_delta=0.03)
    buf       = GestureBuffer(size=12)
    sentence  = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps       = 0

    print("\n=== Gesture Recognition (Right Hand) ===")
    print("SPACE     → Speak sentence")
    print("BACKSPACE → Undo last word")
    print("C         → Clear")
    print("Q         → Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        current_gesture = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                if label != 'Right':
                    continue  # ignore left hand

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                lm     = hand_lm.landmark
                states = get_finger_states(lm)

                # Shake detection (runs every frame, bypasses buffer)
                if shake_det.update(lm[0].x):
                    sentence.append("Hi")
                    speech.speak("Hi", force=True)
                    print("[Shake]: Hi")
                    buf.reset_last()
                    current_gesture = "Hi"
                else:
                    current_gesture = classify_gesture(states, lm)

        # Stability buffer
        stable = buf.update(current_gesture)

        # Add to sentence when stable and not a repeat
        if stable and stable != buf.last_added:
            sentence.append(stable)
            buf.last_added = stable
            print(f"[Added]: {stable}")

        if not stable:
            buf.reset_last()

        # Draw UI
        draw_ui(frame, stable, sentence, speech.is_speaking, fps)
        cv2.imshow("Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if sentence:
                full = " ".join(sentence)
                print(f"[Speaking]: {full}")
                speech.speak(full, force=True)
        elif key == 8:  # BACKSPACE
            if sentence:
                removed = sentence.pop()
                buf.reset_last()
                print(f"[Removed]: {removed}")
        elif key == ord('c'):
            sentence.clear()
            buf.reset_last()
            print("[Cleared]")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
