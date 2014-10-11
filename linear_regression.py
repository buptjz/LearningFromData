# -*- coding: utf-8 -*-
__author__ = 'wangjz'

"""
Learning From Data
HW 1
In these problems, we will explore how Linear Regression for classification works.
As with the Perceptron Learning Algorithm in Homework # 1, you will create your own target function f
and data set D. Take d = 2 so you can visualize the problem, and assume X = [−1, 1] × [−1, 1] with uniform probability
of picking each x ∈ X . In each run, choose a random line in the plane as your target function f (do this by taking
two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the line passing through them), where one
side of the line maps to +1 and the other maps to −1. Choose the inputs xn of the data set as
random points (uniformly in X ), and evaluate the target function on each xn to get the corresponding output yn.
"""

import numpy as np
import random
from perceptron import Perceptron


def avg(a_list):
    return 1.0 * sum(a_list) / len(a_list)


class LinearRegression:
    def __init__(self, training_x, training_y, test_x=None, test_y=None):
        """
        training_x =[[---x1--],
                     [---x2--],
                     [.......],
                     [---xN--]]

        training_y = [y1,y2.....,yN]
        """
        self.__N = training_x.shape[0]
        self.__X = training_x
        self.__Y = training_y
        if test_x != None:
            self.__N_test = test_x.shape[0]
            self.__X_test = test_x
            self.__Y_test = test_y
        self.w = 0

    def learn_w(self):
        X = self.__X
        # Xt = X.transpose()
        #pseudo inverse
        X_pinv = np.linalg.pinv(X)
        self.w = np.dot(X_pinv, self.__Y)

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


def generate_train_and_test(train_num, test_num):
    total = train_num + test_num
    w, X, Y = generate_date(total)
    X_train = X[0:train_num, :]
    X_test = X[train_num:total, :]
    Y_train = Y[0:train_num, :]
    Y_test = Y[train_num:total, :]
    return w, X_train, Y_train, X_test, Y_test


def main2():
    """
    generate 1000 fresh points and use them to estimate the out-of-sample error
    Eout of g that you got in Problem 5 (number of misclassified out-of-sample points / total number of
    out-of-sample points).
    Again, run the experiment 1000 times and take the average. Which value is closest to the average Eout?
    """
    EXP_TIMEs = 1000
    train_num = 100
    test_num = 1000
    E_in_list = []
    E_out_list = []
    for e in range(EXP_TIMEs):
        w, X_train, Y_train, X_test, Y_test = generate_train_and_test(train_num, test_num)
        lr = LinearRegression(X_train, Y_train, X_test, Y_test)
        lr.learn_w()
        E_in_list.append(lr.evaluate_error_in())
        E_out_list.append(lr.evaluate_error_out())

    print(avg(E_in_list))
    print(avg(E_out_list))


def main1():
    """
    Now, take N = 10. After finding the weights using Linear Regression,
    use them as a vector of initial weights for the Perceptron Learning Algorithm. Run PLA until it converges
    to a final vector of weights that completely separates all the in-sample points. Among the choices below,
    what is the closest value to the average number of iterations (over 1000 runs) that PLA takes to converge?
    (When implementing PLA,
    have the algorithm choose a point randomly from the set of misclassified points at each iteration)
    """
    EXP_TIMEs = 1000
    train_num = 10
    iterations_list = []
    for e in range(EXP_TIMEs):
        w, X_train, Y_train = generate_date(train_num)
        lr = LinearRegression(X_train, Y_train)
        lr.learn_w()
        learned_w = lr.w
        #def __init__(self, training_X, training_Y, init_w=[]):
        per = Perceptron(X_train, Y_train, learned_w)
        per.gd_algorithm()
        iterations_list.append(per.num_iterations)
    print(avg(iterations_list))


def nonlinear_exp():
    """
    In these problems, we again apply Linear Regression for classification.
    Consider the target function:
    f(x1,x2)=sign(x21 +x2 −0.6)
    Generate a training set of N = 1000 points on X = [−1, 1] × [−1, 1] with uniform
    probability of picking each x ∈ X . Generate simulated noise by flipping the sign of the
    output in a random 10% subset of the generated training set.
    """


if __name__ == "__main__":
    nonlinear_exp()