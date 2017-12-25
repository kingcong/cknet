
from test.testCases_v3 import L_model_forward_test_case_2hidden
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
X, parameters = L_model_forward_test_case_2hidden()

A,caches = mlp.fprop(X, parameters)
print("A:"+str(A))
print("caches:"+str(caches))

