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

model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data], epochs=1)
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])

priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])
