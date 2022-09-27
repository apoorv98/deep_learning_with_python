#!/usr/bin/env python3

# common imports
from tensorflow.keras.datasets import mnist
import numpy as np

# adding white noise channels and all-zeros channels to mnist
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

train_images_with_noise_channels = np.concenate(
    [train_images, np.random.random((len(train_images), 784))],
    axis=1
)

train_images_with_zeros_channels = np.concenate(
    [train_images, np.zeros((len(train_images), 784))],
    axis=1
)

# training with previous model
from tensorflow import keras
from tensorflow.keras import layers

def get_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = get_model()
history_noise = model.fit(
    train_images_with_noise_channels, train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

model = get_model()
history_zeros = model.fit(
    train_images_with_zeros_channels, train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
