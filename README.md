# Hand Gesture Recognition — University Project
## Build Guide & Documentation

---

## Quick Start

```bash
# 1. Install dependencies
pip install opencv-python mediapipe pyttsx3 numpy

# 2. Run each step in order to build understanding
python step2_landmarks.py      # Test webcam + landmark detection
python step3_classifier.py     # Test gesture classification
python step4_tts.py            # Test text-to-speech
python step5_final_app.py      # Run the full app
```

---

## How It Works

### The Pipeline

```
Webcam Frame
     ↓
MediaPipe Hands  →  21 landmark coordinates (x, y, z)
     ↓
Finger state logic  →  [extended / curled] per finger
     ↓
Gesture classifier  →  gesture label ("Hello", "Stop" …)
     ↓
Stability buffer  →  holds N frames before confirming
     ↓
  ┌──┴──┐
Text   Speech
 UI    (pyttsx3)
```

---

## Gestures Recognised

| Gesture | Hand shape | Output word |
|---|---|---|
| Open palm | All 5 fingers extended | Hello |
| Fist | All fingers curled | Stop |
| Thumbs up | Only thumb extended | Good |
| Peace / Victory | Index + middle extended | Peace |
| Point | Only index extended | Next |
| Pinky up | Only pinky extended | Done |
| Three fingers | Index + middle + ring | Three |
| Rock on | Index + pinky extended | Rock |
| OK | Thumb + middle + ring + pinky | OK |

---

## Controls (Final App)

| Key | Action |
|---|---|
| `SPACE` | Speak the full accumulated sentence aloud |
| `BACKSPACE` | Remove the last word from the sentence |
| `C` | Clear the sentence |
| `Q` | Quit |

---

## How to Add Your Own Gestures

Open `step5_final_app.py` and find the `classify_gesture()` function.

Each gesture is just a pattern of True/False:
```python
# finger_states = [thumb, index, middle, ring, pinky]
#                  True = extended, False = curled

# Example: Add "Love" gesture (index + pinky + thumb extended)
if t and i and not m and not r and p:
    return "Love"
```

---

## Project Extension Ideas (for a better grade)

### 1. ML-based classifier (advanced)
Instead of if-else rules, record training data and train a model:
```python
# Collect: record landmark coordinates + label to a CSV
# Train:   scikit-learn SVM or a small Keras neural network
# Predict: replace classify_gesture() with model.predict()
```

### 2. ISL (Indian Sign Language) support
Map ISL alphabet handshapes to letters, then build words.

### 3. Sentence buffer UI
Build a proper on-screen keyboard that accumulates gestures into full sentences.

### 4. Two-hand detection
Set `max_num_hands=2` in MediaPipe and process both hands independently.

### 5. Export / logging
Save gesture history to a file with timestamps.

---

## Understanding MediaPipe Landmarks

```
       8   12  16  20     ← Fingertips (tip IDs: 4,8,12,16,20)
       |   |   |   |
       7   11  15  19
       |   |   |   |
       6   10  14  18
       |   |   |   |
   4   5   9   13  17     ← Base knuckles (MCP)
   |    \  |   |  /
   3     \ |   | /
   |      \|   |/
   2       0              ← Wrist (landmark 0)
   |
   1
```

Each landmark has `.x`, `.y`, `.z` coordinates (0.0 to 1.0).
- Smaller `y` = higher on screen
- Smaller `z` = closer to camera

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Webcam not opening | Change `VideoCapture(0)` to `VideoCapture(1)` |
| No speech on Linux | Run `sudo apt-get install espeak` |
| Gestures flickering | Increase `STABILITY_FRAMES` in step5 (try 15) |
| Wrong hand detected | Add handedness check: `results.multi_handedness` |
| Low FPS | Lower webcam resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` |

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8–3.11 | Language |
| OpenCV | 4.x | Webcam capture, frame rendering |
| MediaPipe | 0.10.x | Hand landmark detection |
| pyttsx3 | 2.90 | Offline text-to-speech |
| NumPy | 1.x | Numerical helpers |
