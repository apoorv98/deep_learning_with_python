#!/usr/bin/env python3

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


# driver code
model = get_mnist_model()
model.compile(optimizer="rsmprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", RootMeanSquaredError()])
model.fit(train_images, train_labels,
          epochs=3,
          validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)
