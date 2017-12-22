import numpy as np

class Activation():

    def __call__(self, input):
        raise NotImplementedError

    def bprop(self, input, output, above):
        raise NotImplementedError


class Identity(Activation):

    def __call__(self, incoming):
        return incoming

    def bprop(self, input, output, above):
        delta = np.ones(input.shape).astype(float)
        return delta * above


class Sigmoid(Activation):

    def __call__(self, incoming):
        return 1 / (1 + np.exp(-incoming))

    def bprop(self, input, output, above):
        delta = output * (1 - output)
        return delta * above


class Relu(Activation):

    def __call__(self, incoming):
        return np.maximum(incoming, 0)

    def bprop(self, input, output, above):
        delta = np.greater(input, 0).astype(float)
        return delta * above