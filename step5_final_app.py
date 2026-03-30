"""
STEP 5 — Final Integrated App
================================
Full gesture recognition system with:
  ✓ Live webcam with hand landmark overlay
  ✓ Rule-based gesture classifier
  ✓ Stability buffer (no flickering)
  ✓ Text-to-speech output (background thread)
  ✓ On-screen sentence builder (accumulate gestures into phrases)
  ✓ Clean UI overlay

CONTROLS:
  Q         → Quit
  SPACE     → Speak the current sentence aloud and clear it
  BACKSPACE → Remove last word from sentence
  C         → Clear sentence

Gesture → Word mapping:
  Open palm   → "Hello"
  Fist        → "Stop"
  Thumbs up   → "Good"
  Peace       → "Peace"
  Point up    → "Next"
  Pinky up    → "Done"
  Three       → "Three"
  Rock 🤘     → "Rock"
  OK          → "OK"
"""

import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  GESTURE CLASSIFIER  (same logic as Step 3)
# ══════════════════════════════════════════════════════════════════════

TIPS = [4, 8, 12, 16, 20]
PIP  = [3, 6, 10, 14, 18]


def get_finger_states(landmarks):
    states = []
    # Thumb (x-axis comparison)
    states.append(landmarks[4].x < landmarks[3].x)
    # Other fingers (y-axis comparison: tip above pip = extended)
    for tip_id, pip_id in zip(TIPS[1:], PIP[1:]):
        states.append(landmarks[tip_id].y < landmarks[pip_id].y)
    return states   # [thumb, index, middle, ring, pinky]


def classify_gesture(finger_states):
    t, i, m, r, p = finger_states

    if t and i and m and r and p:
        return "Hello"
    if not t and not i and not m and not r and not p:
        return "Stop"
    if t and not i and not m and not r and not p:
        return "Good"
    if not t and i and m and not r and not p:
        return "Shanaya"
    if not t and i and not m and not r and not p:
        return "Next"
    if not t and not i and not m and not r and p:
        return "Done"
    if not t and i and m and r and not p:
        return "Three"
    if not t and i and not m and not r and p:
        return "Rock"
    if t and not i and m and r and p:
        return "OK"
    return None   # Unknown gesture


# ══════════════════════════════════════════════════════════════════════
#  SPEECH ENGINE
# ══════════════════════════════════════════════════════════════════════

class SpeechQueue:
    def __init__(self, cooldown=2.5):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 155)
        self.engine.setProperty('volume', 1.0)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        self.cooldown = cooldown
        self.last_spoken = {}
        self.lock = threading.Lock()
        self.is_speaking = False

    def speak(self, text, force=False):
        """
        Speak text. force=True bypasses cooldown (for sentence playback).
        """
        now = time.time()
        with self.lock:
            if not force and text in self.last_spoken:
                if now - self.last_spoken[text] < self.cooldown:
                    return
            if self.is_speaking:
                return
            self.last_spoken[text] = now
            self.is_speaking = True

        def _speak():
            self.engine.say(text)
            self.engine.runAndWait()
            with self.lock:
                self.is_speaking = False

        threading.Thread(target=_speak, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
#  UI DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════

FONT       = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (80, 220, 100)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (160, 160, 160)
COLOR_ACCENT = (80, 180, 255)
COLOR_BG     = (20, 20, 20)


def draw_rounded_rect(frame, x1, y1, x2, y2, color, alpha=0.6, radius=12):
    """Draw a semi-transparent rounded rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    # Corners
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, stable_gesture, sentence, is_speaking, fps):
    h, w = frame.shape[:2]

    # ── Top bar: gesture display ────────────────────────────────
    draw_rounded_rect(frame, 0, 0, w, 75, COLOR_BG, alpha=0.75)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                FONT, 0.5, COLOR_GRAY, 1)

    # Current gesture
    gesture_display = stable_gesture if stable_gesture else "—"
    cv2.putText(frame, "Gesture:", (15, 28), FONT, 0.6, COLOR_GRAY, 1)
    cv2.putText(frame, gesture_display, (100, 28),
                FONT, 0.7, COLOR_GREEN, 2)

    # Speaking indicator
    if is_speaking:
        cv2.putText(frame, "◉ Speaking", (15, 58),
                    FONT, 0.55, COLOR_ACCENT, 1)
    else:
        cv2.putText(frame, "○ Ready", (15, 58),
                    FONT, 0.55, COLOR_GRAY, 1)

    # ── Bottom bar: sentence builder ───────────────────────────
    bar_h = 80
    draw_rounded_rect(frame, 0, h - bar_h, w, h, COLOR_BG, alpha=0.8)

    sentence_str = " ".join(sentence) if sentence else "(no words yet)"
    # Truncate if too long to display
    max_chars = 50
    if len(sentence_str) > max_chars:
        sentence_str = "..." + sentence_str[-(max_chars - 3):]

    cv2.putText(frame, "Sentence:", (15, h - bar_h + 22),
                FONT, 0.55, COLOR_GRAY, 1)
    cv2.putText(frame, sentence_str, (110, h - bar_h + 22),
                FONT, 0.65, COLOR_WHITE, 2)

    # Controls hint
    controls = "SPACE: Speak  |  BACKSPACE: Undo  |  C: Clear  |  Q: Quit"
    cv2.putText(frame, controls, (15, h - 15),
                FONT, 0.42, COLOR_GRAY, 1)

    # ── Gesture guide (right side) ──────────────────────────────
    guide = [
        ("Open palm", "Hello"),
        ("Fist",      "Stop"),
        ("Thumb up",  "Good"),
        ("Peace",     "Shanaya"),
        ("Point",     "Next"),
        ("Pinky",     "Done"),
    ]
    draw_rounded_rect(frame, w - 195, 80, w - 5, 80 + len(guide) * 28 + 16,
                      COLOR_BG, alpha=0.65)
    for idx, (gesture_name, word) in enumerate(guide):
        y = 100 + idx * 28
        cv2.putText(frame, gesture_name, (w - 188, y),
                    FONT, 0.45, COLOR_GRAY, 1)
        cv2.putText(frame, f"→ {word}", (w - 88, y),
                    FONT, 0.45, COLOR_GREEN, 1)


# ══════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands_detector = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.6
    )

    # TTS
    speech = SpeechQueue(cooldown=2.5)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    STABILITY_FRAMES = 10          # frames gesture must hold to confirm
    gesture_buffer   = []
    stable_gesture   = None
    last_added_word  = None        # word last added to sentence
    sentence         = []          # accumulated words

    # FPS tracking
    prev_time = time.time()
    fps = 0

    print("\n=== Gesture Recognition App ===")
    print("Controls:")
    print("  SPACE     → Speak the sentence")
    print("  BACKSPACE → Remove last word")
    print("  C         → Clear sentence")
    print("  Q         → Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        current_gesture = None

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                states = get_finger_states(hand_lm.landmark)
                current_gesture = classify_gesture(states)

        # ── Stability buffer ─────────────────────────────────────
        gesture_buffer.append(current_gesture)
        if len(gesture_buffer) > STABILITY_FRAMES:
            gesture_buffer.pop(0)

        # Confirm gesture when stable
        if len(gesture_buffer) == STABILITY_FRAMES:
            counts = {}
            for g in gesture_buffer:
                if g:
                    counts[g] = counts.get(g, 0) + 1
            if counts:
                top_gesture, top_count = max(counts.items(), key=lambda x: x[1])
                if top_count >= STABILITY_FRAMES - 2:
                    stable_gesture = top_gesture
                else:
                    stable_gesture = None
            else:
                stable_gesture = None

        # ── Auto-add word to sentence ────────────────────────────
        # Add word when gesture is stable AND it's different from the
        # last added word (so holding the gesture doesn't spam)
        if stable_gesture and stable_gesture != last_added_word:
            sentence.append(stable_gesture)
            last_added_word = stable_gesture
            # Speak individual word
            speech.speak(stable_gesture)

        # Reset last_added_word when no gesture detected
        if not stable_gesture:
            last_added_word = None

        # ── Draw UI ───────────────────────────────────────────────
        draw_ui(frame, stable_gesture, sentence, speech.is_speaking, fps)

        cv2.imshow("Gesture Recognition", frame)

        # ── Keyboard controls ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):       # SPACE: speak full sentence
            if sentence:
                full = " ".join(sentence)
                print(f"[Speaking sentence]: {full}")
                speech.speak(full, force=True)

        elif key == 8:              # BACKSPACE: remove last word
            if sentence:
                removed = sentence.pop()
                print(f"[Removed]: {removed}")
                last_added_word = None

        elif key == ord('c'):       # C: clear
            sentence.clear()
            last_added_word = None
            print("[Cleared sentence]")

    cap.release()
    cv2.destroyAllWindows()
    print("App closed.")


if __name__ == "__main__":
    main()
