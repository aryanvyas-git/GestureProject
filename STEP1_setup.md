# Step 1 — Setup & Installation

## Folder structure
Create this folder structure on your machine:

```
gesture_project/
├── step2_landmarks.py       ← webcam + landmark detection (test)
├── step3_classifier.py      ← gesture classifier (test)
├── step4_tts.py             ← text + speech output (test)
├── step5_final_app.py       ← full integrated app
└── gestures.py              ← gesture definitions (shared)
```

---

## Prerequisites

Make sure you have **Python 3.8–3.11** installed.
Check with:
```bash
python --version
```

---

## Install dependencies

Run this in your terminal inside your project folder:

```bash
pip install opencv-python mediapipe pyttsx3 numpy
```

### What each library does:
| Library | Purpose |
|---|---|
| `opencv-python` | Captures webcam feed, draws on frames |
| `mediapipe` | Detects hand landmarks (21 points per hand) |
| `pyttsx3` | Offline text-to-speech (no internet needed) |
| `numpy` | Math for angle calculations |

---

## Verify installation

Run this quick check:
```python
import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
print("All libraries installed successfully!")
```

Save it as `check.py` and run:
```bash
python check.py
```

If you see the success message, you're ready for Step 2.

---

## Notes
- If `pyttsx3` fails on Linux, also run: `sudo apt-get install espeak`
- On Windows, pyttsx3 uses the built-in SAPI5 engine — no extra install needed
- On Mac, it uses the `say` command — also built-in
