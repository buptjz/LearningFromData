# -*- coding: utf-8 -*-
__author__ = 'wangjz'

import math

'''
For Questions 12-18, implement the AdaBoost-Stump algorithm as introduced in Lecture 208. Run the algorithm on the
following set for training: hw2_adaboost_train.dat and the following set for testing: adaboost_test.dat
Use a total of T=300 iterations (please do not stop earlier than 300), and calculate Ein and Eout with the 0/1 error.

For the decision stump algorithm, please implement the following steps. Any ties can be arbitrarily broken.
1. For any feature i, sort all the xn,i values to x[n],i such that x[n],i≤x[n+1],i.
2. Consider thresholds within −∞ and all the midpoints x[n],i+x[n+1],i2. Test those thresholds with s∈{−1,+1} to
determine the best (s,θ) combination that minimizes Euin using feature i.
3. Pick the best (s,i,θ) combination by enumerating over all possible i.

For those interested, Step 2 can be carried out in O(N) time only!!
'''


class DecisionModel:
    '''
    h_{s,i,theta}(x) = s · sign(xi - theta)
    '''

    def setThetaMinimum(self):
        self.theta = -999999.0

    def __init__(self, theta=-999999.0, thetaIndex=0, s=0, alpha=0, i=0, ein=999999.0):
        self.theta = theta
        self.thetaIndex = thetaIndex
        self.s = s#pos or neg
        self.alpha = alpha#weight of this model
        self.i = i#dimension of this model
        self.ein = ein


def calculateUTeack(train_data):
    N = len(train_data)
    pos_l_u = [0 for _ in range(N)]
    pos_r_u = [0 for _ in range(N)]
    neg_l_u = [0 for _ in range(N)]
    neg_r_u = [0 for _ in range(N)]
    pos_accum = 0
    neg_accum = 0
    for i in range(0, N):
        if train_data[i][LABEL] == "1":
            pos_accum += train_data[i][U_WEIGHT]
        else:
            neg_accum += train_data[i][U_WEIGHT]
        pos_l_u[i] = pos_accum
        neg_l_u[i] = neg_accum

    pos_accum = 0
    neg_accum = 0
    for i in range(N - 1, -1, -1):
        if train_data[i][LABEL] == "1":
            pos_accum += train_data[i][U_WEIGHT]
        else:
            neg_accum += train_data[i][U_WEIGHT]
        pos_r_u[i] = pos_accum
        neg_r_u[i] = neg_accum
    return pos_l_u, pos_r_u, neg_l_u, neg_r_u


LABEL = 2
U_WEIGHT = 3
DIR_POS = 1
DIR_NEG = -1
T = 300  # total iterations
TRAIN_FILE = "./data/hw2_adaboost_train.dat"
TEST_FILE = "./data/hw2_adaboost_test.dat"


def main():
    f_train = open(TRAIN_FILE)
    lines = f_train.readlines()
    f_train.close()

    N = len(lines)
    D = 2
    print 'There are totally [%d] trianing items' % N
    print 'Input has dimension [%d]' % D

    train_data = []

    for l in lines:
        itms = l[:-1].split()
        # (d1,d2,label,u)
        a_list = [float(itms[0]), float(itms[1]), str(itms[2]), 1.0 / N]
        train_data.append(a_list)

    for t in range(T):
        print ""
        print "Iteration: %d" % (t + 1,)
        dm = DecisionModel()

        for d in range(D):
            # keep track of the u weight accumulate from l to r , r to l,
            train_data = sorted(train_data, key=lambda a: a[d])

            (pos_l_u, pos_r_u, neg_l_u, neg_r_u) = calculateUTeack(train_data)

            total_u = pos_r_u[0] + neg_r_u[0]
            print "sum_u = %f" % total_u
            ''' From pos (including pos) to end,consider two situations:
                (1)all pos
                (2)all neg
            '''
            #situation 1
            for position in range(N):
                error_u = neg_r_u[position]
                if not position == 0:
                    error_u += pos_l_u[position]
                e_in = error_u / total_u
                if dm.ein > e_in:
                    dm.ein = e_in
                    dm.thetaIndex = position
                    dm.s = DIR_POS
                    dm.i = d
                    if position == 0:
                        dm.setThetaMinimum()
                    else:
                        dm.theta = (train_data[position][d] + train_data[position - 1][d]) * 0.5

            #situation 2
            for position in range(N):
                error_u = pos_r_u[position]
                if not position == 0:
                    error_u += neg_l_u[position]
                # print error_u,
                e_in = error_u / total_u
                if dm.ein > e_in:
                    dm.ein = e_in
                    dm.thetaIndex = position
                    dm.s = DIR_NEG
                    dm.i = d
                    if position == 0:
                        dm.setThetaMinimum()
                    else:
                        dm.theta = (train_data[position][d] + train_data[position - 1][d]) * 0.5

        # calculate bala_t and alpha
        print "episilon %f" % dm.ein
        bala_t = math.sqrt((1.0 - dm.ein) / dm.ein)
        dm.alpha = math.log(bala_t)

        #adjust Us
        for i in range(N):
            #incorrect: u = u * bala_t
            if (train_data[i][dm.i] >= dm.theta and train_data[i][LABEL] == "-1") or \
                    (train_data[i][dm.i] < dm.theta and train_data[i][LABEL] == "1"):
                train_data[i][U_WEIGHT] *= bala_t

            #correct : u = u / bala_t
            else:
                train_data[i][U_WEIGHT] /= bala_t

        if t == 1:
            for td in train_data:
                print td[U_WEIGHT]
        # print "dim: %d, e_in:%f" % (dm.i, dm.ein,)
        print dm.ein


if __name__ == '__main__':
    main()