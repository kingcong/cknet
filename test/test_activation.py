from cknet.layer import *
from test.testCases_v3 import *
from cknet.activation import *

def test_sigmoid():
    A, W, b = linear_forward_test_case()
    activation = LinearLayer(10, activation=Sigmoid)