"""
STEP 4 — Text-to-Speech Output
================================
This file tests the TTS engine separately before integrating it.

pyttsx3 runs offline (no internet needed).
Speech runs in a background thread so it doesn't freeze the webcam.

Run this file directly to hear the engine speak some test phrases.
"""

import pyttsx3
import threading
import time


# ─── TTS Engine Setup ──────────────────────────────────────────────

def build_tts_engine():
    """
    Creates and configures a pyttsx3 TTS engine.
    Call this once and reuse the returned engine.
    """
    engine = pyttsx3.init()

    # Set speech rate (default ~200 wpm, we slow it slightly)
    engine.setProperty('rate', 160)

    # Set volume (0.0 to 1.0)
    engine.setProperty('volume', 1.0)

    # List available voices and pick one
    voices = engine.getProperty('voices')
    print(f"\nAvailable voices ({len(voices)} found):")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v.name}  —  {v.id}")

    # Try to use voice index 1 (often a female voice on most systems)
    # Change the index if you prefer a different one
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    
    print("\nTTS engine ready.\n")
    return engine


# ─── Thread-safe speech queue ──────────────────────────────────────
# We run TTS in a thread so it doesn't block the video loop

class SpeechQueue:
    """
    Manages speech so:
    1. It doesn't interrupt itself mid-word
    2. It doesn't block the main (video) thread
    3. It won't say the same phrase again within a cooldown period
    """
    def __init__(self, engine, cooldown_seconds=2.0):
        self.engine = engine
        self.cooldown = cooldown_seconds
        self.last_spoken = {}       # phrase → timestamp
        self.lock = threading.Lock()
        self.is_speaking = False

    def speak(self, text):
        """
        Speak `text` in a background thread.
        Won't repeat the same phrase within the cooldown window.
        """
        now = time.time()
        
        with self.lock:
            # Cooldown check — avoid spamming the same word
            if text in self.last_spoken:
                if now - self.last_spoken[text] < self.cooldown:
                    return   # Too soon to repeat

            if self.is_speaking:
                return       # Already talking, skip

            self.last_spoken[text] = now
            self.is_speaking = True

        # Speak in background thread
        def _speak():
            self.engine.say(text)
            self.engine.runAndWait()
            with self.lock:
                self.is_speaking = False

        t = threading.Thread(target=_speak, daemon=True)
        t.start()


# ─── Test the TTS engine ───────────────────────────────────────────

if __name__ == "__main__":
    engine = build_tts_engine()
    speech = SpeechQueue(engine, cooldown_seconds=1.5)

    test_phrases = [
        "Hello! This is the gesture recognition system.",
        "Good.",
        "Stop.",
        "Peace.",
        "Next.",
        "Done.",
    ]

    print("Speaking test phrases...\n")
    for phrase in test_phrases:
        print(f"  Speaking: '{phrase}'")
        speech.speak(phrase)
        time.sleep(2.0)   # Wait between phrases during test

    print("\nTTS test complete. Ready for Step 5.")
