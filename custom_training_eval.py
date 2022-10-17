#!/usr/bin/env python3

from tensorflow.keras.datasets import mnist

def get_mnist_model():
    """model creation factored as function to reuse it later"""
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model


(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

#model = get_mnist_model()
#model.compile(optimizer="rmsprop",
#              loss="sparse_categorical_crossentropy",
#              metrics=["accuracy"])
#
#model.fit(train_images, train_labels,
#          epochs=3,
#          validation_data=(val_images, val_labels))
#
#test_metrics = model.evaluate(test_images, test_labels)
#predictions = model.predict(test_images)

import tensorflow as tf

# subclass the metric class
class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init(name=name, **kwargs)

        # define state variables in constructor
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    # implement state update logic in update_state()
    def update_state(self, y_true, y_pred, sample_weight=None):
        """metric has internal state stored in tf variable
        unlike layers these are not updated via backpropagation
        this method implements the updation logic"""
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        """return the current value of metric"""
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))


    def reset_state(self):
       """method to reset the metric state
       without having to reinstantiate it
       this enables the same metric to be used
       across different epochs of training or
       across both training and evaluation"""
       self.mse_sum.assign(0.)
       self.total_samples.assign(0)


#model = get_mnist_model()
#model.compile(optimizer="rmsprop",
#              loss="sparse_categorical_crossentropy",
#              metrics=["accuracy", RootMeanSquaredError()])
#model.fit(train_images, train_labels,
#          epochs=3,
#          validation_data=(val_images, val_labels))
#test_metrics = model.evaluate(test_images, test_labels)


# using callbacks argument in fit() method
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]


model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", RootMeanSquaredError()])
model.fit(train_images, train_labels,
          epochs=10,
          callbacks=callbacks_list,
          validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)


# custom callback by subclassing the Callback class
from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)),
                 self.per_batch_losses,
                 label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []
