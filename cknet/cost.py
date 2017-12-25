import numpy as np

class Cost():

    def __call__(self, AL, Y):
        raise NotImplementedError

    def bprop(self, AL, Y):
        raise NotImplementedError

class CrossEntropy(Cost):

    def __call__(self, AL, Y):
        m = Y.shape[1]
        cost = -(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T)) / m

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())
        return cost

    def bprop(self, AL, Y):
        dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        assert (dAL.shape == AL.shape)
        return dAL

class SquaredError(Cost):

    def __call__(self, prediction, target):
        return (prediction - target) ** 2 / 2

    def delta(self, prediction, target):
        return prediction - target