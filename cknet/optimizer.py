
class Optimizer():

    def __init__(self, name=None):
        super(Optimizer, self).__init__(name=name)

    def optimizer(self, layer_list, epoch):
        raise NotImplementedError

class GradientDescent(Optimizer):

    def __init__(self, learning_rate, name="gradientDescent"):
        super(GradientDescent, self).__init__(name)
        self.learning_rate = learning_rate

    def optimizer(self, layer_list, epoch):
        raise NotImplementedError

class RMSProp(Optimizer):
    def __init__(self, decay_rate=0.5, learning_rate=2e-3, epsilon=1e-6, name="rmsProp"):
        super(RMSProp, self).__init__(name)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def optimizer(self, layer_list, epoch):
        raise NotImplementedError