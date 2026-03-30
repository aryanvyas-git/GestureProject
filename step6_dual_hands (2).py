"""
DUAL HAND Gesture Recognition — Sentence Builder
=================================================
Left hand  → Subject words  (WHO)  : I, You, We, They, Please, Help
Right hand → Action words   (WHAT) : Need, Want, Stop, Good, Yes, No, Come, Go, More, Done

Sentence builds automatically as you hold gestures.
Press SPACE to speak the full sentence, BACKSPACE to undo, C to clear.

GESTURES:
  LEFT HAND              RIGHT HAND
  Open palm  → I         Open palm  → Need
  Fist       → You       Fist       → Stop
  Thumbs up  → We        Thumbs up  → Good
  Peace      → They      Peace      → Yes
  Point      → Please    Point      → No
  Pinky      → Help      Pinky      → More
  Three      → Come      Three      → Want
  Rock       → We all    Rock       → Go
  OK         → Everyone  OK         → Done
"""

import cv2
import mediapipe as mp
import pyttsx3
import threading
import time


# ══════════════════════════════════════════════════════════════════════
#  GESTURE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

TIPS = [4, 8, 12, 16, 20]
PIP  = [3, 6, 10, 14, 18]


def get_finger_states(landmarks, hand_label):
    """
    Returns [thumb, index, middle, ring, pinky] as True/False.
    hand_label: 'Left' or 'Right' — thumb logic flips per hand.
    """
    states = []

    # Thumb: compare x-axis (flipped for left hand)
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    if hand_label == 'Right':
        states.append(thumb_tip.x < thumb_ip.x)
    else:
        states.append(thumb_tip.x > thumb_ip.x)

    # Other 4 fingers: tip above PIP = extended
    for tip_id, pip_id in zip(TIPS[1:], PIP[1:]):
        states.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return states


# ── Special gesture detectors using z-axis and orientation ────────────

def is_index_pointing_at_camera(landmarks):
    """
    Index finger pointing INTO the camera (toward you).
    Detected when: index is extended AND its tip z is significantly
    more negative (closer to camera) than its base knuckle z.
    """
    tip  = landmarks[8]
    mcp  = landmarks[5]
    pip  = landmarks[6]
    # Tip must be extended (not curled down)
    if not (tip.y < pip.y):
        return False
    # Tip z much smaller (closer) than MCP z = pointing at camera
    return (mcp.z - tip.z) > 0.06


def is_index_rotating(landmarks, hand_label):
    """
    Index finger pointing up and curling/rotating — detected as
    index extended while hand is tilted sideways (wrist rolled).
    We check: index extended AND hand roll (middle finger MCP vs
    index finger MCP on y-axis shows tilt).
    """
    tip   = landmarks[8]
    pip   = landmarks[6]
    # Index must be extended
    if not (tip.y < pip.y):
        return False
    # Other fingers curled
    for tip_id, pip_id in zip(TIPS[2:], PIP[2:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False
    # Hand tilt: wrist z vs middle-MCP z — rotating hand tilts wrist
    wrist      = landmarks[0]
    index_mcp  = landmarks[5]
    middle_mcp = landmarks[9]
    # When hand rotates, index MCP and middle MCP spread apart on x-axis
    horizontal_spread = abs(index_mcp.x - middle_mcp.x)
    return horizontal_spread > 0.07


def is_thumb_pointing_sideways(landmarks, hand_label):
    """
    Thumb extended horizontally AWAY from camera.
    Detected when: only thumb extended AND thumb tip z is close to
    or greater than wrist z (thumb pointing sideways, not up/down).
    Also thumb should be roughly horizontal (tip.y close to base.y).
    """
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist     = landmarks[0]

    # Other fingers must be curled
    for tip_id, pip_id in zip(TIPS[1:], PIP[1:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False

    # Thumb tip and base at similar y = horizontal
    vertical_diff = abs(thumb_tip.y - thumb_mcp.y)
    if vertical_diff > 0.08:
        return False

    # Thumb tip must be clearly separated from wrist on x-axis
    horizontal_reach = abs(thumb_tip.x - wrist.x)
    return horizontal_reach > 0.12


def is_thumb_index_open_horizontal(landmarks, hand_label):
    """
    Thumb AND index finger spread open horizontally (like an L shape
    or measuring gesture) — "minor" gesture.
    Both thumb and index extended, others curled,
    AND they spread wide apart horizontally.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    index_pip = landmarks[6]

    # Index must be extended
    if not (index_tip.y < index_pip.y):
        return False

    # Middle, ring, pinky must be curled
    for tip_id, pip_id in zip(TIPS[2:], PIP[2:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False

    # Thumb and index spread wide apart horizontally
    horizontal_spread = abs(thumb_tip.x - index_tip.x)
    # Also check they are at similar height (both horizontal-ish)
    vertical_diff = abs(thumb_tip.y - index_tip.y)
    return horizontal_spread > 0.15 and vertical_diff < 0.12


def classify_gesture(finger_states, hand_label, landmarks):
    """
    Maps finger states (and special orientation checks) to a word.
    Different vocabulary for left vs right hand.
    landmarks is passed in for the special depth/orientation gestures.
    """
    t, i, m, r, p = finger_states

    if hand_label == 'Right':

        # ── Special orientation-based gestures (check FIRST) ──────
        # These override the simple finger-state gestures

        # Index pointing INTO camera → "this"
        if not t and i and not m and not r and not p:
            if is_index_pointing_at_camera(landmarks):
                return "this"

        # Index rotating (only index up, hand tilted) → "is"
        if not t and i and not m and not r and not p:
            if is_index_rotating(landmarks, hand_label):
                return "is"

        # Thumb sideways away from camera → "my"
        if is_thumb_pointing_sideways(landmarks, hand_label):
            return "my"

        # Thumb + index spread horizontal (L-shape wide) → "minor"
        if is_thumb_index_open_horizontal(landmarks, hand_label):
            return "minor"

        # ── Standard finger-state gestures ────────────────────────
        # Fist → "project" (replaces old "Stop")
        if not t and not i and not m and not r and not p:
            return "project"

        if t and i and m and r and p:                     return "Hi"
        if t and not i and not m and not r and not p:     return "Good"
        if not t and i and m and not r and not p:         return "Yes"
        if not t and i and not m and not r and not p:     return "No"
        if not t and not i and not m and not r and p:     return "More"
        if not t and i and m and r and not p:             return "Want"
        if not t and i and not m and not r and p:         return "Go"
        if t and not i and m and r and p:                 return "Done"

   
    return None


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
        self.lock = threading.Lock()
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
#  SHAKE DETECTOR
# ══════════════════════════════════════════════════════════════════════

class ShakeDetector:
    """
    Detects a horizontal wrist shake on the right hand.
    Looks for rapid direction reversals in the wrist x position.
    """
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
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors (BGR)
C_WHITE  = (245, 245, 245)
C_GRAY   = (160, 160, 160)
C_BLACK  = (10, 10, 10)
C_GREEN  = (80, 220, 100)
C_BLUE   = (255, 180, 80)    # left hand color (blue-ish)
C_ORANGE = (60, 160, 255)    # right hand color (orange-ish)
C_ACCENT = (220, 220, 80)    # sentence text


def draw_panel(frame, x1, y1, x2, y2, alpha=0.72):
    """Semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, left_gesture, right_gesture, sentence, is_speaking, fps):
    h, w = frame.shape[:2]

    # ── Top bar ────────────────────────────────────────────────────
    draw_panel(frame, 0, 0, w, 80)

    # FPS
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 22),
                FONT, 0.5, C_GRAY, 1)

    # Speaking indicator
    status = "◉ Speaking..." if is_speaking else "○ Ready"
    status_color = C_GREEN if is_speaking else C_GRAY
    cv2.putText(frame, status, (w - 90, 45), FONT, 0.45, status_color, 1)

    # Left hand gesture
    cv2.putText(frame, "LEFT:", (15, 30), FONT, 0.6, C_GRAY, 1)
    lg = left_gesture if left_gesture else "—"
    cv2.putText(frame, lg, (80, 30), FONT, 0.75, C_BLUE, 2)

    # Right hand gesture
    cv2.putText(frame, "RIGHT:", (15, 62), FONT, 0.6, C_GRAY, 1)
    rg = right_gesture if right_gesture else "—"
    cv2.putText(frame, rg, (90, 62), FONT, 0.75, C_ORANGE, 2)

    # ── Sentence bar (bottom) ──────────────────────────────────────
    draw_panel(frame, 0, h - 85, w, h)

    sentence_str = " ".join(sentence) if sentence else "(make a gesture to start...)"
    # Truncate if too long
    if len(sentence_str) > 55:
        sentence_str = "..." + sentence_str[-52:]

    cv2.putText(frame, "Sentence:", (15, h - 60), FONT, 0.55, C_GRAY, 1)
    cv2.putText(frame, sentence_str, (15, h - 32),
                FONT, 0.8, C_ACCENT, 2)

    controls = "SPACE: Speak  |  BACKSPACE: Undo  |  C: Clear  |  Q: Quit"
    cv2.putText(frame, controls, (15, h - 10), FONT, 0.4, C_GRAY, 1)

    # ── Left hand gesture guide ────────────────────────────────────
    left_guide = [
        ("Open palm", "I"),
        ("Fist",      "You"),
        ("Thumb up",  "We"),
        ("Peace",     "They"),
        ("Point",     "Please"),
        ("Pinky",     "Help"),
    ]
    gx = 10
    gy = 95
    draw_panel(frame, gx, gy, gx + 185, gy + len(left_guide) * 26 + 28)
    cv2.putText(frame, "LEFT HAND", (gx + 8, gy + 18),
                FONT, 0.48, C_BLUE, 1)
    for idx, (name, word) in enumerate(left_guide):
        y = gy + 38 + idx * 26
        cv2.putText(frame, name, (gx + 8, y), FONT, 0.42, C_GRAY, 1)
        cv2.putText(frame, f"→ {word}", (gx + 110, y), FONT, 0.42, C_BLUE, 1)

    # ── Right hand gesture guide ───────────────────────────────────
    right_guide = [
        ("Shake wrist",   "Hi"),
        ("Point at cam",  "this"),
        ("Index rotate",  "is"),
        ("Thumb sideways","my"),
        ("Thumb+idx wide","minor"),
        ("Fist",          "project"),
        ("Open palm",     "Thank You"),
        ("Thumb up",      "Good"),
    ]
    rx = w - 195
    ry = 95
    draw_panel(frame, rx, ry, rx + 185, ry + len(right_guide) * 26 + 28)
    cv2.putText(frame, "RIGHT HAND", (rx + 8, ry + 18),
                FONT, 0.48, C_ORANGE, 1)
    for idx, (name, word) in enumerate(right_guide):
        y = ry + 38 + idx * 26
        cv2.putText(frame, name, (rx + 8, y), FONT, 0.42, C_GRAY, 1)
        cv2.putText(frame, f"→ {word}", (rx + 110, y), FONT, 0.42, C_ORANGE, 1)


# ══════════════════════════════════════════════════════════════════════
#  STABILITY BUFFER — per hand
# ══════════════════════════════════════════════════════════════════════

class GestureBuffer:
    """
    Smooths out gesture detection — only confirms a gesture
    after it's been held consistently for N frames.
    """
    def __init__(self, size=12):
        self.size = size
        self.buffer = []
        self.stable = None
        self.last_added = None   # last word added to sentence

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
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    # MediaPipe — now tracking 2 hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands_detector = mp_hands.Hands(
        max_num_hands=2,                  # ← KEY CHANGE: track both hands
        min_detection_confidence=0.75,
        min_tracking_confidence=0.6
    )

    speech = SpeechEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Per-hand buffers
    left_buf  = GestureBuffer(size=12)
    right_buf = GestureBuffer(size=12)
    shake_det = ShakeDetector(min_reversals=3, window=18, min_delta=0.03)

    sentence = []

    prev_time = time.time()
    fps = 0

    print("\n=== Dual Hand Gesture Recognition ===")
    print("Left hand  → Subject words (I, You, We, They, Please, Help...)")
    print("Right hand → Action words  (Need, Stop, Good, Yes, No, More...)")
    print("\nControls:")
    print("  SPACE     → Speak the sentence")
    print("  BACKSPACE → Undo last word")
    print("  C         → Clear")
    print("  Q         → Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        # Reset detected gestures each frame
        detected = {'Left': None, 'Right': None}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Get hand label ('Left' or 'Right')
                # We flip the frame horizontally, so MediaPipe's labels
                # already match what the user sees — no swap needed.
                label = handedness.classification[0].label

                # Draw skeleton — different color per hand
                conn_style = mp_drawing_styles.get_default_hand_connections_style()
                lm_style   = mp_drawing_styles.get_default_hand_landmarks_style()
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    lm_style, conn_style
                )

                # Classify
                states  = get_finger_states(hand_lm.landmark, label)
                gesture = classify_gesture(states, label, hand_lm.landmark)
                detected[label] = gesture

                # Shake detection on right hand
                if label == 'Right':
                    wrist_x = hand_lm.landmark[0].x
                    if shake_det.update(wrist_x):
                        sentence.append("Hi")
                        speech.speak("Hi", force=True)
                        print("[Shake detected]: Hi")

        # ── Update buffers & build sentence ───────────────────────
        left_stable  = left_buf.update(detected['Left'])
        right_stable = right_buf.update(detected['Right'])

        # Add to sentence when gesture is stable and different from last added
        for stable, buf in [(left_stable, left_buf), (right_stable, right_buf)]:
            if stable and stable != buf.last_added:
                sentence.append(stable)
                buf.last_added = stable
                # No word-by-word speech — press SPACE to speak full sentence

        # Reset last_added when hand disappears
        if not left_stable:
            left_buf.reset_last()
        if not right_stable:
            right_buf.reset_last()

        # ── Draw UI ───────────────────────────────────────────────
        draw_ui(frame, left_stable, right_stable, sentence,
                speech.is_speaking, fps)

        cv2.imshow("Dual Hand Gesture Recognition", frame)

        # ── Keyboard controls ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if sentence:
                full = " ".join(sentence)
                print(f"[Speaking]: {full}")
                speech.speak(full, force=True)
        elif key == 8:   # BACKSPACE
            if sentence:
                removed = sentence.pop()
                print(f"[Removed]: {removed}")
                left_buf.reset_last()
                right_buf.reset_last()
        elif key == ord('c'):
            sentence.clear()
            left_buf.reset_last()
            right_buf.reset_last()
            print("[Cleared]")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
