from test.testCases_v3 import L_model_backward_test_case
from cknet.model import Model
from cknet.layer import LinearLayer
from cknet.activation import Relu, Sigmoid
from cknet.initializers import RandomInit

layers = [
    LinearLayer(size=20, activation=Relu, input_dim=12288),
    LinearLayer(size=7, activation=Relu),
    LinearLayer(size=5, activation=Sigmoid),
]

mlp = Model(layers=layers, initializer=RandomInit)
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
