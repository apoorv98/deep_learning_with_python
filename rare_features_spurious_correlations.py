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
