from examples.minist.tf_utils import *
from cknet.layer import LinearLayer
from cknet.model import Model
from cknet.activation import Relu,Softmax
from cknet.cost import *
from cknet.optimizer import GradientDescent
from cknet.initializers import RandomInit

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
print("X shape %s, Y shape %s"%(str(X_train_orig.shape),str(Y_train_orig.shape)))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

layers = [
    LinearLayer(size=20, activation=Relu, input_dims=12288),
    LinearLayer(size=10, activation=Relu),
    LinearLayer(size=6, activation=Softmax)
]

mlp = Model(layers=layers, initializer=RandomInit)

cost = CrossEntropy()

optimizer = GradientDescent(learning_rate=0.0001)

mlp.fit(X=X_train, Y=Y_train, cost=cost, optimizer=optimizer,num_epochs=3000)

train_accuracy = mlp.eval(X = X_train, Y = Y_train)
print("train accuracy = " + str(train_accuracy))

test_accuracy = mlp.eval(X = X_test, Y = Y_test)
print("test accuracy = " + str(test_accuracy))