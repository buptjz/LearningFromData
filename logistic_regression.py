# -*- coding: utf-8 -*-
__author__ = 'wangjz'

"""
Logistic Regression
In this problem you will create your own target function f (probability in this case)
and data set D to see how Logistic Regression works. For simplicity, we will take f
to be a 0/1 probability, so y is a deterministic function of x.
Take d = 2 so you can visualize the problem, and let X = [−1, 1]×[−1, 1] with uniform
probability of picking each x ∈ X . Choose a line in the plane as the boundary between
f(x) = 1 (where y has to be +1) and f(x) = 0 (where y has to be −1) by taking two
random, uniformly distributed points from X and taking the line passing through
them as the boundary between y = ±1. Pick N = 100 training points at random
from X , and evaluate the outputs yn for each of these points xn.
Run Logistic Regression with Stochastic Gradient Descent to find g, and estimate Eout
(the cross entropy error) by generating a sufficiently large, separate set of points to
evaluate the error. Repeat the experiment for 100 runs with different targets and take
the average. Initialize the weight vector of Logistic Regression to all zeros in each
run. Stop the algorithm when kw(t−1) − w(t)k < 0.01, where w(t) denotes the weight
vector at the end of epoch t. An epoch is a full pass through the N data points (use a
random permutation of 1, 2, · · · , N to present the data points to the algorithm within
each epoch, and use different permutations for different epochs). Use a learning rate
of 0.01.
"""

from math import e, log, sqrt
import numpy as np
import random


class LogisticRegression:
    def __init__(self, training_x, training_y, eta=0.01, test_x=None, test_y=None, hasTest=False):
        if hasTest:
            self.__NT = test_x.shape[0]
            self.__XT = test_x
            self.__YT = test_y

        self.__N = training_x.shape[0]
        self.__D = training_x.shape[1]
        self.__X = training_x
        self.__Y = training_y
        self.__W = np.zeros((self.__D, 1))
        self.__ETA = eta
        self.__orders = [i for i in range(self.__N)]
        self.iterations = 0

    def gd_learn(self):
        while True:
            self.iterations += 1
            new_W = self.sgd()
            dif = self.__W - new_W
            a = sqrt(sum(dif * dif))
            if a < 0.01:
                break
            self.__W = np.copy(new_W)
            # print(self.__W)

    def sgd(self):
        """随机梯度下降算法"""
        random.shuffle(self.__orders)
        new_w = np.copy(self.__W)
        for o in self.__orders:
            delta_e_in = np.zeros((1, self.__D))
            tmp = self.__Y[o, 0] * np.dot(self.__X[o, :], new_w)#Yn wT Xn
            dividor = 1 + pow(e, tmp)
            dividen = self.__Y[o, 0] * self.__X[o, :]
            delta_e_in += (- dividen / dividor)
            new_w -= self.__ETA * delta_e_in.T
        return new_w

    def cal_e_out(self):
        """计算E_out"""
        e_out = 0
        for o in range(self.__NT):
            tmp = - self.__YT[o, 0] * np.dot(self.__XT[o, :], self.__W)
            e_out += log(1 + pow(e, tmp))
        e_out /= self.__NT
        return e_out

    def cal_e_in(self):
        """计算E_in"""
        e_in = 0
        for o in self.__orders:
            tmp = - self.__Y[o, 0] * np.dot(self.__X[o, :], self.__W)
            e_in += log(1 + pow(e, tmp))
        e_in /= self.__N
        return e_in


def generate_date(n):
    #generate target function (represented by vector w)
    p1 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    p2 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    w = [1, 0, 0]
    w[1] = (p2[0] - p1[0]) / (p1[1] - p2[1])
    w[2] = - (p1[0] * w[0] + p1[1] * w[1])
    w = np.array([[w[0]], [w[1]], [w[2]]])

    #generate n random points between [-1,1]X[-1,1]
    X = np.random.rand(n, 3) * 2 - 1
    X[:, 0] = 1

    Y = np.dot(X, w)
    Y[Y >= 0] = 1
    Y[Y < 0] = -1
    Y = np.int16(Y)
    return w, X, Y


def main():
    EXP_TIMEs = 100
    N_DATAPOINTS = 100
    e_outs = []
    for e in range(EXP_TIMEs):
        w, X, Y = generate_date(N_DATAPOINTS * 20)
        X_train = X[:N_DATAPOINTS, :]
        X_test = X[N_DATAPOINTS:, :]
        Y_train = Y[:N_DATAPOINTS, :]
        Y_test = Y[N_DATAPOINTS:, :]
        lr = LogisticRegression(X_train, Y_train, test_x=X_test, test_y=Y_test, hasTest=True)
        lr.gd_learn()
        eout = lr.cal_e_out()
        e_outs.append(eout)
        print eout,
        print lr.cal_e_in()
    print sum(e_outs) / len(e_outs)

main()