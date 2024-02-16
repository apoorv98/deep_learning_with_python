#!/usr/bin/env python3

def layer_normalization(batch_of_sequences):
    mean = np.mean(batch_of_seqences, keepdims=True, axis=-1)
    variance = np.var(batch_of_sequences, keepdims=True, axis=-1)
    return (batch_of_sequences - mean) / variance
