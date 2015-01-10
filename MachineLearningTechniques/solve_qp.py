# -*- coding: utf-8 -*-
__author__ = 'wangjz'

from math import pow
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp

'''
Solving Question 3
Consider the same training data set as Question 2, but instead of explicitly transforming the input space X to Z,
apply the hard-margin support vector machine algorithm with the kernel function
K(x,x′)=(1+xTx′)2,
which corresponds to a second-order polynomial transformation. Set up the optimization problem using (α1,⋯,α7)
and numerically solve for them
(you can use any package you want). Which of the followings are true about the optimal α?
'''

'''
K(x,x′)=(1+xTx′)2,
'''
def cus_kernel(v1, v2):
    return pow((1 + dot(v1, v2)), 2)

# Problem data.
N = 7
X = matrix([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [-1.0, 0.0],
    [0.0, 2.0],
    [0.0, -2.0],
    [-2.0, 0.0]
    ])

#构造y 列向量
y = matrix([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

#构造Q，就是这里的P
Q = matrix(0.0, (N, N))
for n in range(N):
    for m in range(N):
        Q[n,m] = y[n] * y[m] * cus_kernel(X[:, n], X[:, m])

#构造q，就是这里的q
pbar = matrix(-1.0, (N, 1))

#构造A，就是这里的-G
A = matrix(0.0, (N+2, N))
for ind in range(N):
    A[ind, ind] = 1.0
A[N, :] = y.trans()
A[N+1, :] = -1 * y.trans()
A = -A

#构造c，就是这里的h
c = matrix(0.0, (N + 2, 1))

#不需要等式约束，所以没有这里的A和b

#求解
res = qp(Q, pbar, A, c)['x']
print res
print sum(res)