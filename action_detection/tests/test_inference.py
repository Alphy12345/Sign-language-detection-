import os
from utils import build_synthetic_dataset
import numpy as np
from tensorflow import keras


def test_training_and_inference(tmp_path):
    # tiny smoke test: train model on synthetic data and run one predict
    X, y = build_synthetic_dataset(num_classes=2, samples_per_class=5, timesteps=32)
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 33*4)),
        keras.layers.Conv1D(32,3,padding='same',activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.LSTM(32),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X, y, epochs=1, batch_size=4)
    p = model.predict(X[:1])
    assert p.shape == (1,2)
