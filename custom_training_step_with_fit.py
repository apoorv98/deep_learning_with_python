#!/usr/bin/env python3

loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        self.compiled_metrics.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [loss_tracker]


# driver code
inputs = keras.Input(shape=(28 * 28,))
features = layers.Dense(512, activation="relu")(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation="softmax")(features)

model = CustomModel(inputs, outputs)
model.compile(optimizer=keras.optimizers.RMSprop)
model.fit(train_images, train_labels, epochs=3)
