"""Run inference with a trained model on a video or webcam.
This script extracts keypoints using MediaPipe and runs the Keras model to predict classes.
"""
import argparse
import numpy as np
import cv2
from tensorflow import keras
from utils import video_to_keypoint_sequence


def main(args):
    model = keras.models.load_model(args.weights)
    seq = video_to_keypoint_sequence(args.video, max_frames=args.timesteps, skip=args.skip)
    # model expects batch dim
    pred = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]))
    cls = np.argmax(pred, axis=1)[0]
    print('Predicted class:', cls, 'conf', float(np.max(pred)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='path to video file')
    parser.add_argument('--weights', type=str, required=True, help='path to h5 weights')
    parser.add_argument('--timesteps', type=int, default=64)
    parser.add_argument('--skip', type=int, default=1)
    args = parser.parse_args()
    main(args)
