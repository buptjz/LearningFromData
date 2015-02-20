__author__ = 'wangjz'

import numpy as np
import matplotlib.pyplot as plt

'''
For verifying HW4 Q6
'''

def f1(x_plus, x_minus, x_input):
    w = 2 * (x_plus - x_minus)
    b = - (x_plus**2).sum() + (x_minus**2).sum()
    if (w * x_input).sum() + b > 0:
        return 1
    else:
        return -1

def f2(x_plus, x_minus, x_input):
    w = 2 * (x_minus - x_plus)
    b = (x_plus**2).sum() - (x_minus**2).sum()
    if (w * x_input).sum() + b > 0:
        return 1
    else:
        return -1

def f3(x_plus, x_minus, x_input):
    w = 2 * (x_minus - x_plus)
    b = - (x_plus * x_minus).sum()
    if (w * x_input).sum() + b > 0:
        return 1
    else:
        return -1

def f4(x_plus, x_minus, x_input):
    w = 2 * (x_plus - x_minus)
    b = (x_plus * x_minus).sum()
    if (w * x_input).sum() + b > 0:
        return 1
    else:
        return -1


def show_region(x1, x2, fn):
    #[SHOW] Plot the decision boundary and training points
    start = -10
    end = 10
    npoints = 500
    xrange = np.linspace(start, end, npoints)
    yrange = np.linspace(start, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)

    X = np.array([xgrid.reshape(npoints * npoints), ygrid.reshape(npoints * npoints)]).T
    labels = [fn(x1, x2, x) for x in X]
    zgrid = np.array(labels).reshape(npoints, npoints)
    plt.figure()
    plt.title('Decision Tree')
    plt.pcolor(xgrid, ygrid, zgrid)
    plt.plot(x1[0], x1[1],'ro')
    plt.plot(x2[0], x2[1],'go')
    plt.show()


if __name__ == '__main__':
    x1 = np.array([0,-2])
    x2 = np.array([-1,3])
    show_region(x1, x2, f1)
