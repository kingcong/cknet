import numpy as np

class Cost():

    def __call__(self, prediction, target):
        raise NotImplementedError

    def bprop(self, prediction, target):
        raise NotImplementedError

class CrossEntropy(Cost):
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def __call__(self, prediction, target):
        clipped = np.clip(prediction, self.epsilon, 1 - self.epsilon)
        cost = target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
        return -cost

    def bprop(self, prediction, target):
        denominator = np.maximum(prediction - prediction ** 2, self.epsilon)
        delta = (prediction - target) / denominator
        return delta

class SquaredError(Cost):

    def __call__(self, prediction, target):
        return (prediction - target) ** 2 / 2

    def delta(self, prediction, target):
        return prediction - target