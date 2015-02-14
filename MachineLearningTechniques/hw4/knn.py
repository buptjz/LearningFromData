# -*- coding: utf-8 -*-
__author__ = 'wangjz'

import numpy as np
from collections import Counter


def majority_item(Y):
    '''Return the item which occupies the majority in list Y'''
    frequence = Counter(Y)
    tup = max(frequence.items(), key=lambda a:a[1])
    return tup[0]


def knn(train_X, train_Y, predict_X, k=1):
    ntrain = train_X.shape[0]
    npredict = predict_X.shape[0]
    predict_Y = np.empty(npredict)

    for i in range(npredict):
        x = predict_X[i,:]
        diff_mat = np.tile(x, (ntrain, 1)) - train_X
        distances = (diff_mat**2).sum(axis=1)**0.5
        sortidx = np.argsort(distances)#ascent
        k_sortidx = sortidx[:k]
        labels = train_Y[k_sortidx]
        predict_Y[i] = majority_item(labels)
    return predict_Y


def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int)
    return X, Y

def main():
    train_file = "hw4_knn_train.dat"
    test_file = "hw4_knn_test.dat"

    train_X,train_Y = load_file(train_file)
    test_X,test_Y = load_file(test_file)

    print '----------------------------------------'
    print '        Homework 4 Question 15,17       '
    print '----------------------------------------'
    print 'Which of the following is closest to Ein(gnbor)?'
    ks = [1,5]
    for k in ks:
        m = train_X.shape[0]
        pre_Y = knn(train_X, train_Y, train_X, k)
        print float(np.sum(pre_Y != train_Y)) / m

    print '----------------------------------------'
    print '        Homework 4 Question 16,18       '
    print '----------------------------------------'
    print 'Which of the following is closest to Eout(gnbor)?'
    for k in ks:
        mtest = test_X.shape[0]
        test_pre_Y = knn(train_X, train_Y, test_X, k)
        print float(np.sum(test_pre_Y != test_Y)) / mtest


if __name__ == '__main__':
    main()