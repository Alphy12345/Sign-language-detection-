"""Utility functions: keypoint extraction (MediaPipe), dataset helpers, preprocessing."""

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_pose = mp.solutions.pose

pose_ds = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints_from_frame(frame):
    ":""Return a flattened vector of pose keypoints (x,y,z,visibility) for the frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_ds.process(img)
    if not results.pose_landmarks:
        # return zeros for 33 landmarks * 4
        return np.zeros(33*4, dtype=np.float32)
    kp = []
    for lm in results.pose_landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(kp, dtype=np.float32)


def video_to_keypoint_sequence(video_path, max_frames=128, skip=1):
    cap = cv2.VideoCapture(video_path)
    seq = []
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip != 0:
            continue
        kp = extract_keypoints_from_frame(frame)
        seq.append(kp)
        idx += 1
        if idx >= max_frames:
            break
    cap.release()
    seq = np.array(seq)
    # pad if shorter
    if seq.shape[0] < max_frames:
        pad = np.zeros((max_frames - seq.shape[0], seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return seq


def build_synthetic_dataset(num_classes=3, samples_per_class=10, timesteps=64):
    """Create a tiny synthetic dataset for smoke testing/training quickly."""
    X = []
    y = []
    for c in range(num_classes):
        for s in range(samples_per_class):
            # simple deterministic synthetic patterns per class
            base = np.sin(np.linspace(0, 3.14*(c+1), timesteps))
            noise = np.random.normal(0, 0.05, (timesteps, 33*4))
            pattern = (base.reshape(timesteps,1) * (np.ones((1,33*4))))
            seq = pattern + noise
            X.append(seq.astype(np.float32))
            y.append(c)
    X = np.stack(X)
    y = np.array(y)
    return X, y
