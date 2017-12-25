import numpy as np

class Layer():

    def __init__(self, name=None):
        super(Layer, self).__init__(name)
        self.name = name


class LinearLayer(Layer):

    def __init__(self, size, activation, input_dims = 0):
        self.size = size
        self.activation = activation()
        self.input_dims = input_dims

    def fprop(self, A_prev, W, b):

        Z, linear_cache = self.linear_prop(A_prev, W, b)
        A, activation_cache = self.activation(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def linear_prop(self, A, W, b):
        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def bprop(self, dA, cache):
        linear_cache, activation_cache = cache

        dZ = self.activation.bprop(dA, activation_cache)
        dA_prev, dW, db = self.linear_bprop(dZ, linear_cache)
        return dA_prev, dW, db

    def linear_bprop(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db