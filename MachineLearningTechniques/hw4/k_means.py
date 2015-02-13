# -*- coding: utf-8 -*-
__author__ = 'wangjz'

import numpy as np


def sqr_error(x1, x2):
    return ((x1 - x2) ** 2).sum()


def compute_error(data, labels, mu):
    N = data.shape[0]

    err_accum = 0
    for i in range(N):
        err_accum  += sqr_error(data[i, :],mu[labels[i]])

    return err_accum / N


def k_means(k,data):
    '''Basoc K-Means algorithm'''

    N = data.shape[0]
    labels = np.zeros(N).astype(np.int)

    #randomly chosen k from xn to initialize mu
    mu = data[np.random.choice(N, k)].copy()

    nloops = 0
    while True:
        nloops += 1
        converge = True
        #2.1 Update labels (S)
        for i in range(N):
            cur_point = data[i, :]
            diff_mat = np.tile(cur_point, (k, 1)) - mu
            distances = (diff_mat**2).sum(axis=1)**0.5
            min_idx = np.argmin(distances)
            labels[i] = min_idx

        #2.2 Update mu
        for c in range(k):
            idx = (labels == c)
            if np.count_nonzero(idx) != 0:
                points = data[idx, :]
                new_mu = points.sum(axis=0) / points.shape[0]#avg
                if not (new_mu == mu[c, :]).all():
                    converge = False
                    mu[c, :] = new_mu

        if converge:
            break

    return labels, mu


def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
    X = data[:, :-1]
    return X


def main():
    train_file = "hw4_kmeans_train.dat"
    train_X = load_file(train_file)
    T = 500
    ks = [2,10]
    print '----------------------------------------'
    print '        Homework 4 Question 19(20)      '
    print '----------------------------------------'
    for k in ks:
        Eins = []
        for i in range(T):
            labels, mu = k_means(k,train_X)
            Eins.append(compute_error(train_X, labels, mu))
        print 'For k=' + str(k) + ', which of the following is closest to the average Ein of k-Means over 500 experiments?'
        print sum(Eins) / float(len(Eins))


if __name__ == '__main__':
    main()