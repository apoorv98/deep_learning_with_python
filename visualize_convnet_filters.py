#!/usr/bin/env python3

# first, instantiate the Xception model,
# loaded with weights
# pretrained on the ImageNet dataset
model = keras.applications.xception.Xception(
    weights="imagenet",
    include_top=False
)

# we're interested in the convolutional layers of the model
# the Conv2D and SeparableConv2D layers
# we'll need to know their names
# so we can retrieve their outputs
# let's print their names
# in order of depth
for layer in model.layers:
    if isinstance(layer,
                  (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layes.name)


# not let's create a second model
# that returns the output of a specific layer
# aka /a feature extractor/ model
layer_name = "block3_sepconv1"
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

# using the feature extractor
activation = feature_extractor(
    keras.applications.xception.preprocess_input(img_tensor)
)

# let's use our feature extractor model
# to define a function that returns a scalar value
# quantifying how much a given input image "activates"
# a given filter in the layer.
# this is the "loss function"
# that we'll maximize during the gradient ascent process
import tensorflow as tf

def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    ## we avoid border artifacts
    ## by only involving non-border pixels in the loss;
    ## we discard the first two pixels along the sides of activation
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


# loss maximization via stochastic gradient ascent
@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image


# function to generate filter visualizations
img_width = 200
img_height = 200

def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10.
    image = tf.random.uniform(
        minval=0.4,
        maxval=0.6,
        shape=(1, img_width, img_height, 3)
    )

    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)

    return image[0].numpy()


# the resulting image tensor
# is a floating point array of shape(200, 200, 3)
# with value that may not be integers
# hence, we need to post-process this tensor
# to turn it into a displayable image
def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]
    return image


# generating a grid
# of all filter response patterns in a layer
all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image == deprocess_image(
        generate_filter_pattern(filter_index)
    )
    all_images.append(image)

margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]

        row_start = (cropped_width + margin) * i
        row_end = (cropped_width + margin) * i + cropped_width
        column_start = (cropped_height + margin) * j
        column_end = (cropped_height + margin) * j + cropped_height

        stitched_filters[row_start: row_end, column_start: column_end, :] = image

keras.utils.save_img(
    f"filters_for_layer_{layer_name}.png", stitched_filters
)
