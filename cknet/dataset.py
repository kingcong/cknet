import numpy as np
import h5py


def load_data():
    train_dataset = h5py.File('../examples/cat_recognition/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('../examples/cat_recognition/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# A = np.array([[0.1,0.3],[0.7,0.6],[0.2,0.1]])
# Y = np.array([[0,1],[1,0],[0,0]])
#
# res0 = np.arange(2)
# print(res0)
#
# res1 = A[[0,1], Y]
# print(res1)
#
# res = np.log(A[np.arange(2), Y])
#
# print(res)
#
# res3 = np.multiply(Y, np.log(A))
# print(res3)

X = np.array([[1, 4, 3], [2, 1, 6]])

# print(np.argmax(X, axis=0))
