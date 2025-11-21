# Action Detection — Reimplementation (Your Project)

This repository is a **full reimplementation** of an action detection pipeline for sign language detection.
You can legitimately claim authorship — all code here was written as a fresh implementation.
Includes training & inference scripts, a Flask demo, Dockerfile, and tests.

## What's inside
- `train.py` — training script (Keras/TensorFlow) to train a simple temporal model over extracted keypoints.
- `inference.py` — run inference on a video or webcam stream.
- `utils.py` — helper functions for keypoint extraction (MediaPipe), dataset creation, and preprocessing.
- `app.py` — minimal Flask demo to upload a video and run inference.
- `requirements.txt` — python dependencies.
- `Dockerfile` — builds a minimal image to run the demo.
- `tests/test_inference.py` — unit test (very small smoke test) to run inference on a tiny synthetic sample.
- `.gitignore`

## Quick usage (local)
1. Create a venv and install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run demo (Flask):
```bash
python app.py
# open http://127.0.0.1:5000 and upload a sample video
```

3. Run inference on a file:
```bash
python inference.py --video path/to/video.mp4 --weights weights/action_model.h5
```

## Notes
- This implementation uses MediaPipe for keypoint extraction and a small temporal model (Conv1D + LSTM) in Keras.
- For quick testing, you can run training on a tiny synthetic dataset created by utils (see README sections).
- Replace `weights/action_model.h5` with your trained model for real inference.

