

class Layer():

    def __init__(self, name=None):
        super(Layer, self).__init__(name)
        self.name = name


class LinearLayer(Layer):

    def __init__(self, size, activation, init = None, input_dim = 0):
        self.size = size
        self.activation = activation
        self.init = init
        self.input_dim = input_dim