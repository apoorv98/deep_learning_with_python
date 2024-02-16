#!/usr/bin/env python3

# Neural style transfer in Keras

## Listing 12.16 Getting the style and content images

import tensorflow as tf
from tensorflow import keras

base_image_path = keras.utils.get_file(
    "sf.jpg", origin="https://img-datasets.s3.amazonaws.com/sf.jpg"
)
style_reference_image_path = keras.utils.get_file(
    "starry_night.jpg", origin="https://img-datasets.s3.amazonaws.com/starry_night.jpg"
)

original_width, original_height = keras.utils.load_img(base_image_path).size
img_height = 400
img_width = round(original_width * img_height / original_height)


## Listing 12.17 Auxilliary functions

import numpy as np

def preprocess_image(image_path):
    """Util function to open, resize, and format
    images into appropriate arrays"""
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_image(img):
    """Util function to convert
    Numpy array into a valid image"""
    img = img.reshape((img_height, img_width, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


## Listing 12.18 Using a pretrained VGG19 model
## to create a feature extractor

model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(input=model.inputs, outputs=outputs_dict)


## Listing 12.19 Content loss

def content_loss(base_img, combination_img):
    return tf.reduce_sum(tf.square(combination_img - base_img))
