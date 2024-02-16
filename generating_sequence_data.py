#!/usr/bin/env python3

# Listing 12.1 Reweighting a probability distribution
# to a different temperature

import numpy as np

def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


# Listing 12.3 Creating a dataset from text files

import tensorflow as tf
from tensorflow import keras
dataset = keras.utils.text_dataset_from_directory(
    directory="aclImdb", label_mode=None, batch_size=256
)
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))


# Listing 12.4 Preparing a *TextVectorization* layer

from tensorflow.keras.layers import TextVectorization

sequence_length = 100
vocab_size = 15000

text_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
text_vectorization.adapt(dataset)


# Listing 12.5 Setting up a language modelling dataset

def prepare_lm_dataset(text_batch):
    ## convert batch of texts
    ## to a batch of integer sequences
    vectorized_sequences = text_vectorization(text_batch)
    ## create inputs by cutting off the last word
    x = vectorized_sequences[:, :-1]
    ## create targets by offsetting the sequences by 1
    y = vectorized_sequences[:, 1:]



# Listing 12.6 A simple Transformer-based language model

from tensorflow.keras import layers
embed_dim = 256
latent_dim = 2048
num_heads = 2

inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")


# Listing 12.7 The text-generation callback

import numpy as np

## dict that maps word indices back to strings,
## to be used for text decoding
tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))

## implements variable-temperature sampling
## from a probability distribution

def sample_next(predictions, temperature=1.0):
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)


class TextGenerator(keras.callbacks.Callback):
    def __init__(self,
                 prompt,
                 generate_length,
                 model_input_length,
                 temperatures=(1.,),
                 print_freq=1):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        for temperature in self.temperatures:
            print("== Generating with temperature", temperature)
            sentence = self.prompt

            for i in range(self.generate_length):
                tokenized_sentence = text_vectorization([sentence])
                predictions = self.model(tokenized_sentence)
                next_token = sample_next(predictions[0, i, :])
                sampled_token = tokens_index[next_token]
                sentence += " " + sampled_token
            print(sentence)

prompt = "This movie"
text_gen_callback = TextGenerator(
    prompt,
    generate_length=50,
    model_input_length=sequence_length,
    temperatures=(0.2, 0.5, 0.7, 1., 1.5)
)


# Listing 12.8 Fitting the language model
model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])
