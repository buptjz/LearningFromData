import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class DT_Node:
    '''Decision Tree Node'''
    def __init__(self, axis=None, thres=None):
        self.left = None#riht child
        self.right = None#left child
        self.axis = axis#axis is the dimension of this branching
        self.thres = thres#thres is the threshold in that dimension
        self.label = None#class label which determines if it's a leave node


def plot_sign(X, Y):
    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'ro')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'go')


def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int)
    return X, Y


def majority_item(Y):
    '''Return the item which occupies the majority in list Y'''
    frequence = Counter(Y)
    tup = max(frequence.items(), key=lambda a:a[1])
    return tup[0]


def gini_index(Y):
    '''Compute gini index, consider all k class'''
    Y_list = list(Y)
    N = len(Y_list)
    freqs = [Y_list.count(c) for c in set(Y_list)]
    error = 1 - sum([((f*1.0) / N)**2 for f in freqs])
    return error


def decide_branch(X, Y):
    '''Decide axis and theta according to the training datas'''
    D, N = X.shape[1], X.shape[0] #num of features,examples
    # init with not branch at all
    min_err = N * gini_index(Y)
    best_axis = 0
    best_thres = -np.inf
    #consider different features
    for d in range(D):
        # sort the rows by feature (column) d
        sortidx = X[:, d].argsort()
        X_sorted = X[sortidx]
        Y_sorted = Y[sortidx]
        #consider different branching positions (totally N-1)
        for i in range(N-1):
            #split between i and i+1,left #i+1 examples and right #N-i-1 examples
            # error = size(D1) * impurity(D1)+size(D2) * impurity(D2)
            err = (i+1) * gini_index(Y_sorted[:i+1]) +\
                       (N-i-1) * gini_index(Y_sorted[i+1:]) #gini_err = left + right brach gini
            if min_err > err:
                min_err = err
                best_axis = d
                best_thres = (X_sorted[i, d] + X_sorted[i+1, d]) * 0.5

    return best_axis, best_thres


def datas_above(X, Y, axis, thres):
    # warning : shallow copy
    reserved = X[:, axis] > thres
    return X[reserved, :], Y[reserved]


def datas_below(X, Y, axis, thres):
    reserved = X[:, axis] < thres
    return X[reserved, :], Y[reserved]


def all_same_rows(X):
    #check if all xs are the same
    for d in range(X.shape[1]):
        if not set(X[:, d]) == 1:
            return False
    return True


def build_decision_tee(X, Y):
    '''
    Building decision tree recursively
    datas: a list of datas
    '''
    assert X.shape[0] != 0

    #base case 1: all yn the same: impurity = 0,gt(x) = yn
    if len(set(Y)) == 1:
        leave_node = DT_Node()
        leave_node.label = Y[0]
        return leave_node
    #all xn the same: no decision stumps, gt(x) = majority of yn
    elif all_same_rows(X) :
        leave_node = DT_Node()
        leave_node.label = majority_item(Y)
        return leave_node
    #common case
    else:
        axis, thres = decide_branch(X, Y)
        node = DT_Node(axis, thres)
        left_X, left_Y = datas_below(X, Y, axis, thres)
        right_X, right_Y = datas_above(X, Y, axis, thres)
        node.left = build_decision_tee(left_X, left_Y)
        node.right = build_decision_tee(right_X, right_Y)
        return node


def predict_y(dtree, x):
    '''Given a decision tree, predict the label of x'''
    assert dtree
    if dtree.label:
        return dtree.label
    else:
        if x[dtree.axis] < dtree.thres:
            return predict_y(dtree.left, x)
        else:
            return predict_y(dtree.right, x)


def print_tree(dtree, spaces):
    global num_nodes

    indendent = spaces * "\t"
    if dtree.label:#leave node
        print "%s+%d" % (indendent, dtree.label)
        return
    else:
        num_nodes += 1
        print "%s%f(%d)" % (spaces * "\t", dtree.thres, dtree.axis)
    print_tree(dtree.left, spaces + 2)
    print_tree(dtree.right, spaces + 2)


num_nodes = 0
if __name__ == '__main__':
    train_file = "hw3_train.dat"
    test_file = "hw3_test.dat"
    X_train, Y_train = load_file(train_file)
    N = X_train.shape[0]# num of training examples
    dtree = build_decision_tee(X_train, Y_train)

    #Q13 : How many internal nodes (branching functions) are there in the resulting tree G?
    print_tree(dtree, 0)
    print '----------------------------------------'
    print '         Homework 3 Question 13         '
    print '----------------------------------------'
    print 'How many internal nodes:'
    print num_nodes

    #Q14 : Which of the following is closest to the Ein (evaluated with 0/1 error) of the tree?
    Predicts_train = [predict_y(dtree, x)for x in X_train]
    num_wrong = N - sum(Predicts_train == Y_train)
    print '----------------------------------------'
    print '         Homework 3 Question 14         '
    print '----------------------------------------'
    print 'Ein (evaluated with 0/1 error):'
    print  float(num_wrong) / N

    #Q15 : Which of the following is closest to the Eout (evaluated with 0/1 error) of the tree?
    X_test, Y_test = load_file(test_file)
    Predicts_test = [predict_y(dtree, x)for x in X_test]
    N = X_test.shape[0]
    num_wrong = N - sum(Predicts_test == Y_test)
    print '----------------------------------------'
    print '         Homework 3 Question 15         '
    print '----------------------------------------'
    print 'Eout (evaluated with 0/1 error):'
    print  float(num_wrong) / N

    #Show the decision boundary and training points
    start = 0
    end = 1
    npoints = 500
    xrange = np.linspace(start, end, npoints)
    yrange = np.linspace(start, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange,yrange)

    X = np.array([xgrid.reshape(npoints * npoints), ygrid.reshape(npoints * npoints)]).T
    labels = [predict_y(dtree,x) for x in X]
    zgrid = np.array(labels).reshape(npoints, npoints)
    plt.figure()
    plt.title('Decision Tree')
    plot_sign(X_train,Y_train)
    plt.pcolor(xgrid, ygrid, zgrid)
    plt.show()
    '''We can see overfit obviously!!'''