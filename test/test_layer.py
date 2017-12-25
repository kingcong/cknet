from cknet.layer import LinearLayer,Layer
from cknet.activation import Relu

layer = LinearLayer(100, activation=Relu, input_dim=100)
print(layer)