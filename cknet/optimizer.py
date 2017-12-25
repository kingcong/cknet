class Optimizer():

    def __init__(self, name=None):
        self.name = name

    def optimizer(self, parameters, grads, epoch):
        raise NotImplementedError

class GradientDescent(Optimizer):

    def __init__(self, learning_rate, name="gradientDescent"):
        super(GradientDescent, self).__init__(name)
        self.learning_rate = learning_rate

    def optimizer(self, parameters, grads, epoch):
        L = len(parameters) // 2
        for i in range(1, L + 1):
            W = parameters["W" + str(i)]
            b = parameters["b" + str(i)]
            dW = grads["dW" + str(i)]
            db = grads["db" + str(i)]
            parameters["W" + str(i)] = W - self.learning_rate * dW
            parameters["b" + str(i)] = b - self.learning_rate * db
        return parameters

class RMSProp(Optimizer):
    def __init__(self, decay_rate=0.5, learning_rate=2e-3, epsilon=1e-6, name="rmsProp"):
        super(RMSProp, self).__init__(name)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def optimizer(self, parameters, grads, epoch):
        raise NotImplementedError