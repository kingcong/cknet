
class Model():

    def __init__(self, layers, dataset=None, name="model", optimizer=None, init=None):
        super(Model, self).__init__()
        self.layers = layers
        self.dataset = dataset
        self.name = name
        self.optimizer = optimizer
        self.epoch_index = 0

    def fit(self, dataset, cost, optimizer, num_epochs, callbacks):

        self.optimizer = optimizer
        self.dataset = dataset
