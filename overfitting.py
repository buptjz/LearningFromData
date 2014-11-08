# -*- coding: utf-8 -*-
__author__ = 'wangjz'

"""
Learning From Data
HW 6
    In the following problems use the data provided in the files
                      http://work.caltech.edu/data/in.dta
                     http://work.caltech.edu/data/out.dta
    as a training and test set respectively. Each line of the files corresponds to a two- dimensional
    input x = (x1,x2), so that X = R2, followed by the corresponding label from Y = {−1, 1}.
    We are going to apply Linear Regression with a non-linear transformation for classification.
    The nonlinear transformation is given by
"""

import numpy as np


class LinearRegression:
    def __init__(self, training_x, training_y, test_x=None, test_y=None, hasTest=False, lam=0.001):
        """
        带weight decay的线性回归
        training_x =[[---x1--],
                     [---x2--],
                     [.......],
                     [---xN--]]
        training_y = [y1,y2.....,yN]
        """
        self.__N = training_x.shape[0]
        self.__X = training_x
        self.__Y = training_y
        if hasTest:
            self.__N_test = test_x.shape[0]
            self.__X_test = test_x
            self.__Y_test = test_y
        self.w = 0
        self.lam = lam

    def learn_w(self):
        X = self.__X
        # Xt = X.transpose()
        #pseudo inverse
        # X_pinv = np.linalg.pinv(X)
        # self.w = np.dot(X_pinv, self.__Y)
        '''wreg = (ZTZ + λI)−1 ZTy'''
        Xt = X.transpose()
        tmp = np.dot(Xt, X) + self.lam * np.identity(8)
        self.w = np.dot(np.dot(np.mat(tmp).I, Xt), self.__Y)

    def evaluate_error_in(self):
        guess = np.dot(self.__X, self.w)
        guess[guess >= 0] = 1
        guess[guess < 0] = -1
        guess = np.int16(guess)
        correct = sum(guess == self.__Y)
        return 1.0 - correct * 1.0 / self.__N

    def evaluate_error_out(self):
        guess = np.dot(self.__X_test, self.w)
        guess[guess >= 0] = 1
        guess[guess < 0] = -1
        guess = np.int16(guess)
        correct = sum(guess == self.__Y_test)
        return 1.0 - correct * 1.0 / self.__N_test


def read_data(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    train_num = len(lines)
    dimension = 8
    X = np.zeros((train_num, dimension))
    Y = np.zeros((train_num, 1))
    for i, l in enumerate(lines):
        items = l[:-1].split()
        X[i, 0] = 1.0                        # 1.0
        X[i, 1] = float(items[0])            # x1
        X[i, 2] = float(items[1])            # x2
        X[i, 3] = X[i, 1] * X[i, 1]          # x1 * x1
        X[i, 4] = X[i, 2] * X[i, 2]          # x2 * x2
        X[i, 5] = X[i, 1] * X[i, 2]          # x1 * x2
        X[i, 6] = abs(X[i, 1] - X[i, 2])     # |x1 - x2|
        X[i, 7] = abs(X[i, 1] + X[i, 2])     # |x1 + x2|
        Y[i, 0] = float(items[2])

    return X, Y


def overfit_exp():
    """
    Run Linear Regression on the training set after performing the non-linear trans- formation.
    What values are closest (in Euclidean distance) to the in-sample and out-of-sample classification errors,
    respectively?
    得到了正确的结果分别是 0.02857143 和 0.084
    """

    trainX, trainY = read_data("in.dta.txt")
    testX, testY = read_data("out.dta.txt")
    lr = LinearRegression(trainX, trainY, testX, testY, True, lam=0.1)
    lr.learn_w()
    print lr.evaluate_error_in()
    print lr.evaluate_error_out()


if __name__ == "__main__":
    overfit_exp()