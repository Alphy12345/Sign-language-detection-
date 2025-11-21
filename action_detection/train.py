"""Train a simple temporal model on extracted keypoints."""
import os
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from utils import build_synthetic_dataset


def build_model(timesteps=64, features=33*4, num_classes=3):
    inp = keras.Input(shape=(timesteps, features))
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    # for quick demo/training we build synthetic dataset
    X, y = build_synthetic_dataset(num_classes=args.num_classes, samples_per_class=args.samples_per_class, timesteps=args.timesteps)
    print("X shape", X.shape, "y shape", y.shape)
    model = build_model(timesteps=args.timesteps, features=33*4, num_classes=args.num_classes)
    model.summary()
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)
    os.makedirs('weights', exist_ok=True)
    model.save(os.path.join('weights', args.output))
    print('Saved model to', os.path.join('weights', args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--samples_per_class', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='action_model.h5')
    args = parser.parse_args()
    main(args)
