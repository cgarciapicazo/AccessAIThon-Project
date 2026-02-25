# AccessAIThon-Project

Real-time **sign-to-speech** prototype: uses a webcam + MediaPipe hand landmarks to classify **static hand signs** (e.g., letters/gestures) with a small PyTorch model, then reads the detected label aloud using TTS.

Current categories include british sign language numbers from 0 to 10 and static letters in the alphabet.

---

## Demo / How to run

### 1) Install dependencies
```bash
pip install -r requirements.txt

### 2) Run the main program (webcam inference)
```bash
python src/main.py
```

Controls:
- Press **Esc** to exit.

---

## How it works (high level)

The application pipeline in `src/main.py` is:

1. **Capture frames** from the webcam (OpenCV).
2. **Detect hands + landmarks** using MediaPipe HandLandmarker.
3. **Convert landmarks → feature tensor (84 floats)**:
   - 21 landmarks × (x,y) × 2 hands = 84
   - Coordinates are normalized (wrist-relative and scale-relative).
4. **Classify** with a lightweight PyTorch MLP (`StaticSignClassifier`).
5. **Stabilize predictions** with an exponential moving average (EMA) over probabilities to reduce flicker.
6. If prediction is **confident and consistent across frames**, show it in an accessible UI overlay and **speak** the label using TTS.

---

## Repository structure

```text
.
├── src/
│   ├── main.py                    # Main entrypoint: webcam → classify → UI + TTS
│   ├── data/
│   │   ├── cache.pt               # Cached tensors + class index mapping used at runtime
│   │   └── examples/
│   │       └── test_datasets.py   # Small helpers to generate toy datasets (demo/dev)
│   ├── models/
│   │   ├── static_network.py      # MLP used for static signs (used by main)
│   │   └── movement_network.py    # Prototype model (currently not used)
│   ├── preprocessing/
│   │   ├── hand_detector.py       # MediaPipe detector wrapper + helpers
│   │   └── tensor_manipulation.py # Landmark result → 84D tensor feature vector
│   ├── postprocessing/
│   │   └── text_to_speech.py      # TTSService (Edge TTS) + audio playback
│   └── utils/
│       └── ema.py                 # Exponential moving average for probabilities
├── training/
│   └── train_static.py            # Script to train/save the static classifier
├── requirements.txt
└── LICENSE
```

---

## Training (static signs)

### `training/train_static.py`
This script trains the `StaticSignClassifier` using images under:
- `src/data/images/<class_name>/*`

It:
- uses MediaPipe to extract landmarks for each image
- converts them to the same **84D** tensor representation used at inference time
- saves a cached dataset to `src/data/cache.pt`
- saves model weights to `src/models/saved_models/static_sign.pth`

Run (example):
```bash
python training/train_static.py
```

> If you add new classes, re-run training so `cache.pt` and the model weights reflect the new label set.

---

## Notes / Troubleshooting

- **Camera not opening**: make sure no other app is using the webcam and that OpenCV has camera permissions.
- **Missing `hand_landmarker.task`**: `src/preprocessing/hand_detector.py` loads it from `src/models/saved_models/hand_landmarker.task`.
- **Audio/TTS issues**: TTS uses `edge_tts` plus `sounddevice`/`soundfile`. On some systems, audio drivers/devices need to be configured.

---

## License
See [LICENSE](LICENSE).