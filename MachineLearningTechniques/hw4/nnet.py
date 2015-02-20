# -*- coding: utf-8 -*-
__author__ = 'wangjz'

import numpy as np
from random import uniform, randint


def arctanh(x):
    return np.arctanh(x)


def tanh(x):
    """comppute tanh(x)"""
    return np.tanh(x)


def d_tanh(x):
    """compute derivative of tanh,
    tanh'(x)"""
    return 1.0 / (np.cosh(x)**2)


def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int)
    return X, Y


def predict_y(x, weights, nnodes):
    nlyrs = len(nnodes)
    Xs = [[]] * nlyrs
    for i in range(1, nlyrs):#exclude 1st layer
        Xs[i] = [1] * nnodes[i]

    #insert x_0 = 1
    x_list = list(x)
    x_list.insert(0, 1)
    Xs[0] = x_list
    for l in range(1, nlyrs):#last layer is output
        for j in range(1, nnodes[l]):#each X_0^l is 1
            Xs[l][j] = compute_x(nnodes, weights, Xs, l, j)

    L = len(nnodes) - 1
    return Xs[L][1]


def compute_x(nnodes, weights, Xs, l, j):
    #Xs[l][j] = x_j^{(3)}
    pre_nnode = nnodes[l-1] #how many nodes in previous layer,include
    s = 0
    for i in range(pre_nnode):
        s += weights[l][i][j] * Xs[l-1][i]
    return tanh(s)


def compute_delta(nnodes, weights, deltas, Xs, l, j, y=0):
    delta = 0
    L = len(nnodes) - 1

    #Special case: l == L
    if l == L:
        s1L = arctanh(Xs[L][1])
        delta = - 2 * (y - Xs[L][1]) * (d_tanh(s1L))

    #general case
    else:
        #k = 1~d^(l+1)
        for k in range(1, nnodes[l+1]):
            sjl = arctanh(Xs[l][j])
            delta += (deltas[l+1][k]) * (weights[l+1][j][k]) * (d_tanh(sjl))

    return delta


def nnet(train_X, train_Y, r, M, eta, T):
    #nodes per layer (d-8-3-1) ->(d+1,9,4,2)
    ntrain = train_X.shape[0]
    nnodes = [i+1 for i in M]
    nnodes.append(2)
    nnodes.insert(0, train_X.shape[1] + 1)

    nlyrs = len(nnodes)

    #just one hidden layers,w[l][i][j] = w_ij^l, first layer is empty
    weights = [[]]

    #Xss[l][i] = X_i^{(l)}, Xss[0] = input
    deltas = [[]] * nlyrs
    Xs = [[]] * nlyrs

    #random init weights
    for idx in range(1, nlyrs):
        i = nnodes[idx-1]
        j = nnodes[idx]
        tmp = []
        for _ in range(i):
            tmp.append([uniform(-r, r) for _ in range(j)])
        weights.append(tmp)

    for i in range(1, nlyrs):#exclude 1st layer
        Xs[i] = [1] * nnodes[i]
        deltas[i] = [0] * nnodes[i]

    for t in range(T):
        # Step 1 : stochastic: randomly pick n ∈ {1, 2, · · · , N}
        choice = randint(0, ntrain-1)
        x_input = train_X[choice, :]
        y_input = train_Y[choice]

        # Step 2 : forward: compute all x(l) with x(0) = xn
        x_list = list(x_input)
        x_list.insert(0, 1)#insert x_0 = 1
        Xs[0] = x_list
        for l in range(1, nlyrs):#last layer is output
            for j in range(1, nnodes[l]):#each X_0^l is 1
                Xs[l][j] = compute_x(nnodes, weights, Xs, l, j)

        # Step 3 : backward: compute all δ(l) subject to x(0) = xn
        reversed_layers = reversed(range(1, nlyrs))
        for l in reversed_layers:
            for j in range(1, nnodes[l]):
                deltas[l][j] = compute_delta(nnodes, weights, deltas, Xs, l, j, y_input)

        # Step 4 : gradient descent: w(l) ← w(l) − ηx(l−1)δ(l)
        for l in range(1, nlyrs):
            for i in range(nnodes[l-1]):
                for j in range(1, nnodes[l]):
                    weights[l][i][j] -= eta * Xs[l-1][i] * deltas[l][j]

    return weights, nnodes


def compute_e_out(test_X, test_Y, weights, nnodes):
    M = test_X.shape[0]
    err_accum = 0.0
    for i in range(M):
        x_input = test_X[i, :]
        y = test_Y[i]
        err_accum += (predict_y(x_input, weights, nnodes) - y) ** 2
    return err_accum / M


def main():
    # Part1 : Load data
    train_file = "hw4_nnet_train.dat"
    test_file = "hw4_nnet_test.dat"
    train_X,train_Y = load_file(train_file)
    test_X,test_Y = load_file(test_file)

    # Part2 : Initialization
    T = 50000#experiment times
    rs = [0.1]#w_ij ranges from (-r,r)
    # Ms = [1, 6, 11, 16, 21]#number of hidden neurons
    Ms = [[8,3]]#number of hidden neurons
    etas = [0.01]#learning rate

    # Part3 : Run experiments
    loops = 500
    for r in rs:
        for M in Ms:
            for eta in etas:
                eouts = []
                print "[Range = %.3f][M =%s][eta = %.3f]" % (r, M, eta)
                for lo in range(loops):
                    weights, nnodes = nnet(train_X, train_Y, r, M, eta, T)
                    eouts.append((compute_e_out(test_X, test_Y, weights, nnodes)))
                print "avg eout = %.4f" % (sum(eouts) / len(eouts))


if __name__ == '__main__':
    main()
