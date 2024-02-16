#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activaiton="relu")(inputs)
# set aside the residual
residual = x

# this is the layer around which we create residual connection
# it increases the number of output filters from 32 to 64
# we use padding="same" to avoid downsampling
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
residual = layers.Conv2D(64, 1)(residual)


# additionally if we have max pooling layer
x = layers.Conv2D(2, padding="same")(x)
residual = layers.Conv2D(64, 1, strides=2)(residual)

# now the block output and residual have same shape
# and can be added
x = layers.add([x, residual])


# Example of simple convnet structured into a series of blocks
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1./255)(inputs)

## Utility function to apply a conv block
## with residual connection
## with an option to add max pooling
def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling(2, padding="same")(x)
        # if we use max pooling,
        # we add a strided convolution
        # to project the residual to the expected shape
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        # if we don't use max pooling,
        # we only project the residual
        # if the number of channels has changed
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x

## First block
x = residual_block(x, filters=32, pooling=True)
## Second block, note the increasing filter count
x = residual_block(x, filters=64, pooling=True)
## the last block doesn't need a max pooling layer
## since we will apply global average pooling right after it
x = residual_block(x, filters=128, pooling=False)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
