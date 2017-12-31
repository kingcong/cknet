import numpy as np
from cknet.regularization import L2Regularization

class Model():

    def __init__(self, layers, name="model", initializer=None):
        super(Model, self).__init__()
        self.layers = layers
        self.name = name
        self.optimizer = None
        self.epoch_index = 0
        self.cost = None
        self.total_cost = []
        self.initializer = initializer()
        self.weights = None
        self.class_number = 0
        self.regularization = L2Regularization(lambd=0.01)

    def init_parameters(self):
        layer1 = self.layers[0]
        input_dims = layer1.input_dims
        layer_dims = [layer.size for layer in self.layers]
        layer_dims.insert(0, input_dims)
        parameters = self.initializer.fill(layer_dims)
        return parameters

    def fit(self, X, Y, cost, optimizer, num_epochs):
        self.optimizer = optimizer
        self.cost = cost

        np.random.seed(1)
        self.weights = self.init_parameters()
        lastlayer = self.layers[len(self.layers)-1]
        self.class_number = lastlayer.size

        for i in range(0, num_epochs):
            self.weights = self.epoch_fit(X, Y, self.weights, i)


    def epoch_fit(self, X, Y, weights, epoch):
        AL, caches = self.fprop(X, weights)
        cost = self.cost(AL, Y)
        # Regularization
        cost += self.regularization.cost_regularization(weights)
        grads = self.bprop(AL, Y, caches)
        weights = self.optimizer.optimizer(weights, grads, epoch)
        self.total_cost.append(cost)

        if epoch % 100 == 0:
            print ("Cost after iteration %i: %f" % (epoch, cost))
        return weights

    def fprop(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L + 1):
            layer = self.layers[l - 1]
            A_prev = A
            A, cache = layer.fprop(A_prev,
                                   parameters["W" + str(l)],
                                   parameters["b" + str(l)])
            caches.append(cache)

        assert (A.shape == (layer.size, X.shape[1]))
        return A, caches

    def bprop(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        # m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = self.cost.bprop(AL, Y)
        grads["dA"+str(L+1)] = dAL

        for l in reversed(range(L)):
            current_cache = caches[l]
            layer = self.layers[l]
            dA_prev_temp, dW_temp, db_temp = layer.bprop(grads["dA"+str(l+2)], current_cache)

            # L2 Regularization
            _, W, _ = current_cache[0]
            dW_temp += self.regularization.brop_regularization(W, L)

            grads["dA"+str(l+1)] = dA_prev_temp
            grads["dW"+str(l+1)] = dW_temp
            grads["db"+str(l+1)] = db_temp
        return grads

    def predict(self, X):

        probas = self._get_result(X)

        if self.class_number == 1:
            return probas > 0.5
        else:
            return np.argmax(probas, axis=0)


    def eval(self, X, Y):
        m = X.shape[1]
        p = self.predict(X)

        if self.class_number == 1:
            accuracy = np.sum((p == Y) / m)
            return accuracy
        else:
            indices_Y = np.argmax(Y, axis=0)
            accuracy = np.sum((p == indices_Y) / m)
            return accuracy



    def _get_result(self, X):
        m = X.shape[1]
        p = np.zeros((self.class_number, m))

        probas, _ = self.fprop(X, self.weights)

        return probas
