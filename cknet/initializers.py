import numpy as np

class Initializer():

    def fill(self, layer_dims):

        raise NotImplementedError()

class RandomInit(Initializer):

    def fill(self, layer_dims):
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
        return parameters