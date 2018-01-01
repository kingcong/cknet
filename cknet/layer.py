import numpy as np

class Layer():

    def __init__(self, name=None):
        super(Layer, self).__init__(name)
        self.name = name

    def fprop(self, A_prev, W, b):
        raise NotImplementedError

    def bprop(self, dA, cache):
        raise NotImplementedError


class LinearLayer(Layer):

    def __init__(self, size, activation, input_dims = 0):
        self.size = size
        self.activation = activation()
        self.input_dims = input_dims

    def fprop(self, A_prev, W, b):

        Z, linear_cache = self.linear_prop(A_prev, W, b)
        A, activation_cache = self.activation(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache, None)

        return A, cache

    def linear_prop(self, A, W, b):
        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def bprop(self, dA, cache):
        linear_cache, activation_cache, _ = cache

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

class Dropout(Layer):

    """
    A dropout layer
    """

    def __init__(self, size, activation, input_dims = 0, keep_prob = 0.5):
        """
        dropout layer init method
        :param size: layer size
        :param activation: layer activation
        :param input_dims: the dimension of first layer
        :param keep_prob: fraction of the inputs that should be stochastically kept
        """
        self.size = size
        self.activation = activation()
        self.input_dims = input_dims
        self.keep_prob = keep_prob


    def fprop(self, A_prev, W, b):
        """
        dropout layer forward propogation method
        :param A_prev: the input of prev layer
        :param W: the parameter of W in current layer
        :param b: the parameter of b in current layer
        :return: the input of current layer and cache(linear_cache and acivation_cache)
                cache for use in backpropogation
        """
        Z, linear_cache = self.linear_prop(A_prev, W, b)
        A, activation_cache = self.activation(Z)

        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < self.keep_prob
        A = A * D
        A = A / self.keep_prob

        d_cache = D

        cache = (linear_cache, activation_cache, d_cache)

        return A, cache

    def bprop(self, dA, cache):
        """
        dropout layer back propogation method
        :param dA: the derivative of dLoss / dA
        :param cache: save cache in forward propogation
        :return: the derivative of dLoss / dX, dLoss / dW, dLoss / db
        """

        linear_cache, activation_cache, d_cache = cache

        dA = dA * d_cache
        dA = dA / self.keep_prob

        dZ = self.activation.bprop(dA, activation_cache)
        dA_prev, dW, db = self.linear_bprop(dZ, linear_cache)

        return dA_prev, dW, db

    def linear_prop(self, A, W, b):
        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

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