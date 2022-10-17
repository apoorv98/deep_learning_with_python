#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers

# listing 7.9
# system to rank customer support tickets py priority
# and route them to appropriate department

vocabulary_size = 10000
num_tags = 100
num_departments = 4

# define model inputs
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# combine input features into a single tensor, by concatenating them
features = layers.Concatenate()([title, text_body, tags])
# apply intermediate layer to get richer representations
features = layers.Dense(64, activation="relu")(features)

# define model outputs
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax", name="department")(features)

# create model by specifying input and output
model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])


# listing 7.10
# training model by providing input and target arrays
import numpy as np

num_samples = 1280

# dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(optimizer="rmsprop", loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])

# using tensorboard to monitor and visualize results
tensorboard = keras.callbacks.TensorBoard(
    log_dir="./logs/",
)

# logs can be viewed using tensorboard --logdir ./logs/

model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data], epochs=1)
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])

priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])


# step-by-step custom training loop: the training step func  Listing 7.21
model = get_mnist_model()

loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()

def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    logs = {}

    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()

    loss_tracking_metric.update_state(loss)

    logs["loss"] = loss_tracking_metric.result()

    return logs


# resetting the metrics
def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()


# step-by-step training loop: the loop itself  Listing 7.23
training_dataset = tf.data.Datasets.from_tensor_slices((train_images, train_labels))
training_dataset = training_dataset.batch(32)
epochs = 3
for epoch in range(epochs):
    reset_metrics()
    for inputs_batch, targets_batch in training_dataset:
        logs = train_step(inputs_batch, targets_batch)
        print(f"Results at the end of epoch {epoch}")

    for key, value in logs.items():
        print(f"...{key}: {value:.4f}")


# and finally the evaluation loop which is just part of training loop defined above
def test_step(inputs, targets):
    predictions = model(inputs, training=False)
    loss = loss_fn(targets, predictions)

    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["val_" + metric.name] = metric.result()
        loss_tracking_metric.update_state(loss)
        logs["val_loss"] = loss_tracking_metric.result()

    return logs


val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()

for inputs_batch, targets_batch in val_dataset:
    logs = test_step(inputs_batch, targets_batch)

print("Evaluation results:")
for key, value in logs.items():
    print(f"...{key}: {value:.4f}")
