import numpy as np

class Activation():

    def __call__(self, input):
        raise NotImplementedError

    def bprop(self, input, output, above):
        raise NotImplementedError

class Sigmoid(Activation):

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