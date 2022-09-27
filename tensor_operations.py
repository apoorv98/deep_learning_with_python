#!/usr/bin/env python3

''' Naive implementation of element-wise tensor relu operation'''
def naive_relu(x):
    assert len(x.shape) == 2    # x is a rank-2 numpy tensor
    x = x.copy()    # avoid overwritting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

''' Naive implementation of element-wise tensor addition'''
def naive_add(x, y):
    assert len(x.shape) == 2    # x and y are rank-2 numpy tensors
    assert x.shape == y.shape
    x = x.copy()    # avoid overwritting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


''' Naive implementation of adding matrix and vector'''
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2    # x is a rank-2 numpy tensor
    assert len(y.shape) == 1    # y is a numpy vector
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


''' Naive implementation of dot product'''
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1    # x and y are numpy vectors
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


''' Naive implementation of matrix and vector dot product'''
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2    # x is a numpy matrix
    assert len(y.shape) == 1    # y is a numpy vector
    assert x.shape[1] == y.shape[0] # matrix multiplication rule (1,32) and (32,)
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


''' Naive matrix dot product'''
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z
