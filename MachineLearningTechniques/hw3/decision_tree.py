import numpy as np
import matplotlib.pyplot as plt

def plot_sign(X, Y):
    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'ro')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'go')

# load_file
# =========
# load a file and get [X Y]
#
def load_file(file_name):
    lines = open(file_name, 'r').readlines()

    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    return X, Y


train_file = "hw3_train.dat"
test_file = "hw3_test.dat"


if __name__ == '__main__':
    X_train, Y_train= load_file(train_file)
    m = X_train.shape[0]    # number of training examples
    d = X_train.shape[1]    # feature dimension

    # let's plot the decision function
    plt.figure()
    plot_sign(X_train,Y_train)
    plt.show()