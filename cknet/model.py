import numpy as np
import math
from cknet.regularization import L2Regularization
from cknet.util import *

class Model():

    def __init__(self, layers, name="model", initializer=None, regularization=L2Regularization,
                 batch_size = 64):
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
        self.regularization = regularization(lambd=0.01)
        self.X = None
        self.Y = None
        self.layer_dims = None

    def init_parameters(self):
        layer1 = self.layers[0]
        input_dims = layer1.input_dims
        layer_dims = [layer.size for layer in self.layers]
        layer_dims.insert(0, input_dims)
        self.layer_dims = layer_dims
        parameters = self.initializer.fill(layer_dims)
        return parameters

    def fit(self, X, Y, cost, optimizer, num_epochs):
        self.optimizer = optimizer
        self.cost = cost

        np.random.seed(1)
        self.weights = self.init_parameters()
        lastlayer = self.layers[len(self.layers)-1]
        self.class_number = lastlayer.size

        seed = 10

        for i in range(0, num_epochs):

            seed = seed + 1
            self.weights = self.epoch_fit(X, Y, self.weights, i, seed=seed)


    def epoch_fit(self, X, Y, weights, epoch, seed = 0):

        minibatches = self.random_mini_batches(X, Y, mini_batch_size=64, seed=seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            AL, caches = self.fprop(minibatch_X, minibatch_Y, weights)

            cost = self.cost(AL, minibatch_Y)

            # Regularization
            cost += self.regularization.cost_regularization(weights)

            grads = self.bprop(AL, minibatch_Y, caches)

            # gradients checking
            # self.gradient_check(weights, grads, X, Y)

            weights = self.optimizer.optimizer(weights, grads, epoch)


        self.total_cost.append(cost)

        if epoch % 100 == 0:
            print ("Cost after iteration %i: %f" % (epoch, cost))
        return weights

    def fprop(self, X, Y, parameters):
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

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):

        """
        Creates a list of random minibatches from (X, Y)

        :param X: input data, of shape (input size, number of examples)
        :param Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        :param mini_batch_size: size of the mini-batches, integer
        :param seed: random seed
        :return: mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

        """

        np.random.seed(seed)
        m = X.shape[1]  # number of training examples
        mini_batches = []

        class_num = Y.shape[0]

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((class_num, m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def gradient_check(self, parameters, gradients, X, Y, epsilon=1e-7):

        """
        gradien checking for back propagation process
        :param parameters: weights of forward propagation
        :param gradients: grads of back propagation
        :param X: input
        :param Y: input label
        :param epsilon: gradient checking parameter
        :return: checking result
        """

        m = len(parameters) // 2
        parameters_values, _ = dictionary_to_vector(parameters)
        grad = gradients_to_vector(gradients, m)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        # Compute gradapprox
        for i in range(1000):
            thetaplus = np.copy(parameters_values)  # Step 1
            thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
            AL, caches = self.fprop(X, Y, vector_to_dictionary(thetaplus, self.layer_dims))
            J_plus[i]= self.cost(AL, Y) # Step 3

            thetaminus = np.copy(parameters_values)  # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
            AL, caches = self.fprop(X, Y, vector_to_dictionary(thetaminus, self.layer_dims))
            J_minus[i]= self.cost(AL, Y)  # Step 3

            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        grad = grad[:1000]
        gradapprox = gradapprox[:1000]
        numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
        # difference = numerator / denominator  # Step 3'
        if denominator == 0:
            difference = 0
        else:
            difference = np.exp(np.log(numerator)-np.log(denominator))

        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

        return difference

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

        probas, _ = self.fprop(X, None, self.weights)

        return probas
