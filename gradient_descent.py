# -*- coding: utf-8 -*-
# __author__ = 'wangjz'

"""
Learning From Data
HW #5

Consider the nonlinear error surface E(u, v) = (uev − 2ve−u
)
2
. We start at the point
(u, v) = (1, 1) and minimize this error using gradient descent in the uv space. Use
η = 0.1 (learning rate, not step size).

How many iterations (among the given choices) does it take for the error E(u, v)
to fall below 10−14 for the first time? In your programs, make sure to use double
precision to get the needed accuracy

w(t + 1) = w(t) − η ∇Ein(w(t))

"""

from math import e


def partial_u(u, v):
    """计算u偏导"""
    fir = 2 * (u * pow(e, v) - 2 * v * pow(e, -u))
    sec = (pow(e, v) + 2 * v * pow(e, -u))

    return fir * sec


def partial_v(u, v):
    """计算v偏导"""
    fir = 2 * (u * pow(e, v) - 2 * v * pow(e, -u))
    sec = (u * pow(e, v) - 2 * pow(e, -u))

    return fir * sec
