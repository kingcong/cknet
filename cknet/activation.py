import numpy as np

class Activation():

    """
    Base class for activation, Child classes can either implement
    the below ``__call__`` and ``bprop`` methods
    """


    def __call__(self, Z):
        """
        Returns the input as output.

        :param input:
            Z (numpy type matrix or array): input value
        :return:
            numpy type matrix or array: identical to input
        """
        raise NotImplementedError

    def bprop(self, dA, cache):
        """
        Return the derivative of A(Z) = Activation(Z)
        :param dA: the derivative of Loss(A), dA = dLoss / dA
        :param cache: the input of Z
        :return: the derivative of Z: dZ = dA / dZ
        """
        raise NotImplementedError

class Sigmoid(Activation):

    """
    Sigmoid activation function, inherit from Activation class, implement
    the method of __call and bprop
    """

    def __call__(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def bprop(self, dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert (dZ.shape == Z.shape)
        return dZ


class Relu(Activation):

    def __call__(self, Z):
        A = np.maximum(0, Z)
        assert (A.shape == Z.shape)
        cache = Z
        return A, cache

    def bprop(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ

class Softmax(Activation):

    def __call__(self, Z):
        cache = Z
        exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True), cache

    def bprop(self, dA, cache):
        return dA