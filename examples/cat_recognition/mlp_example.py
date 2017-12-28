from cknet.layer import LinearLayer
from cknet.model import Model
from cknet.activation import Relu,Sigmoid
from cknet.cost import BinaryCrossEntropy
from cknet.optimizer import GradientDescent
from cknet.dataset import load_data
from cknet.initializers import RandomInit
import matplotlib.pyplot as plt

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

layers = [
    LinearLayer(size=20, activation=Relu, input_dims=12288),
    LinearLayer(size=7, activation=Relu),
    LinearLayer(size=5, activation=Relu),
    LinearLayer(size=1, activation=Sigmoid)
]

mlp = Model(layers=layers, initializer=RandomInit)

cost = BinaryCrossEntropy()

optimizer = GradientDescent(learning_rate=0.01)

mlp.fit(X=train_x, Y=train_y, cost=cost, optimizer=optimizer,num_epochs=1000)

train_accuracy = mlp.eval(X = train_x, Y = train_y)
print("train accuracy = " + str(train_accuracy))

test_accuracy = mlp.eval(X = test_x, Y = test_y)
print("test accuracy = " + str(test_accuracy))

plt.plot(mlp.total_cost)
plt.show()