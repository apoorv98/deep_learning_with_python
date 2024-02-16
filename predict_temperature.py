#!/usr/bin/env python3

# lets look at the data
import os
fname = os.path.join("temperature/jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines)) # this outputs 420,551 lines of data

# now convert all lines of data
# into NumPy arrays
# one array for temperature,
# and another one for rest of the data
# NOTE: we discard the "Date Time column"
import numpy as np
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    # store column 1 in the "temperature" array
    temperature[i] = values[1]
    # we store all columns (including the temperature)
    # in the raw_data array
    raw_data[i, :] = values[:]


# plot the timeseries to observe yearly periodicity
# the data spans 8 years
from matplotlib import pyplot as plt
plt.plot(range(len(temperature)), temperature)

# plotting the first 10 days
# because data is recorded every 10 minutes
# we get 24*6 = 144 data points per day
plt.plot(range(1440), temperature[:1440])


# computing the number of samples
# we'll use for each data split
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)


# each feature in the data is on a different scale
# so we need to normalize the data
# the mean and standard deviation will only be computed
# considering the training data
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std


# 10.7 instantiating datasets
# for training, validation and testing
# Each dataset yields a tuple (samples, targets)
# where samples is a batch of 256 samples
# each containing 120 consecutive hours of input data
# and targets is corresponding array of 256 target temperatures
# NOTE: samples are randomly shuffled,
# so two consecutive samples in a batch
# aren't necessarily temporally close
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)


# Listing 10.8 inspecting the output
# of one of our datasets
for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break


# Common sense, non-machine learning baseline
# Listing 10.9 computing the common-sense
# baseline MAE
def evaluate_naive_method(dataset):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        ## we normalized our features,
        ## so to retrieve a temperature in degree Celsius,
        ## we need to un-normalize it
        ## by multiplying it by std and adding back the mean
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE:{evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE:{evaluate_naive_method(test_dataset):.2f}")


# Basic machine learning model
# it's useful to try simple, cheap machine learning models
# before looking into complicated and computationally expensive models
# this is to make sure any further complexity you throw at the problem
# is legitimate and delivers real benefit
# Listing 10.10 Training and evaluating a densely connected model
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras",
                                    save_best_only=True)
]

model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])

history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

# reload the best model and evaluate it on test data
model = keras.load_model("jena_dense.keras")
print(f"Tests MAE:{model.evaluate(test_dataset)[1]:.2f}")


# Listing 10.11 Plotting results
import matplotlib.pyplot as plt
loss = history.histor["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()


# 1D convolutional model
inputs = keras.Input(shape=(sequence_length,
                            raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 12, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 6, activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckPoint("jena_conv.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.load_model("jena_conv.keras")
print(f"Test MAE:{model.evaluate(test_dataset)[1]:.2f}")


# Listing 10.12 A simple LSTM-based model
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("jena_lstm.keras")
print(f"Test MAE:{model.evaluate(test_dataset)[1]:.2f}")


# Using recurrent dropout to fight overfitting
# Listing 10.22 Training and evaluating
# a dropout-regularized LSTM
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm_dropout.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)


# Listing 10.23 Training and evaluating
# a dropout-regularized, stacked GRU model
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.GRU(32, recurrent_dropout=0.5,
               return_sequences=True)(inputs)
x = layers.GRU(32, recurrent_dropout=0.5)(x)
x = layers.Dropout(0.5)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_stacked_gru_dropout.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model("jena_stacked_gru_dropout.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


# Listing 10.24 Training and evaluating
# a bidirectional LSTM
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Bidirectional(layers.LSTM(16))(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset)
