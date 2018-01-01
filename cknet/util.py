import numpy as np
import operator

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for i in range(len(parameters)):

        if i % 2 == 0:
            key = "W" + str(int(i/2) + 1)
        else:
            key = "b" + str(int(i/2) + 1)

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta, layer_dims):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}

    start = 0
    end = 0

    for i in range(1, len(layer_dims)):
        current_dims = layer_dims[i]
        prev_dims = layer_dims[i-1]
        w_number = current_dims * prev_dims
        b_number = current_dims * 1

        end += w_number
        parameters["W"+str(i)] = theta[start:end].reshape((current_dims,prev_dims))

        start = end
        end = start + b_number
        parameters["b"+str(i)] = theta[start:end].reshape((current_dims, 1))
        start = end

    return parameters


def gradients_to_vector(gradients, m):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    count = 0
    for i in range(m):
        # flatten parameter
        if i % 2 == 0:
            key = "dW" + str(int(i/2) + 1)
        else:
            key = "db" + str(int(i/2) + 1)
        new_vector = np.reshape(gradients[key], (-1, 1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta


def gradient_check_n_test_case():
    np.random.seed(1)
    x = np.random.randn(4, 3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5, 4)
    b1 = np.random.randn(5, 1)
    W2 = np.random.randn(3, 5)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return x, y, parameters

# x, y, parameters = gradient_check_n_test_case()
# print("parameters = \n" + str(parameters))
# vector,_ = dictionary_to_vector(parameters)
# # print("parameters vector = \n" + str(vector))
#
# vector_parameters = vector_to_dictionary(np.copy(vector), [4,5,3,1])
# print("vector parameters = \n" + str(vector_parameters))




