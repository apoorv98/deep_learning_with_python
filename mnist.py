from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

# loading the MNIST dataset in keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# the network architecture
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# the compilation step
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_entropy",
              metrics=["accuracy"])

# preprocess the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


# fitting the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)
