__author__ = 'wangjz'

import numpy as np
import matplotlib.pyplot as plt

def f4(u_plus, u_minus, beta_plus, beta_minus, x_input):
    w = 2 * (beta_plus * u_plus - beta_minus * u_minus)
    b = beta_plus * np.dot(u_plus ,u_plus) - beta_minus * np.dot(u_minus, u_minus)
    return 1 if (w * x_input).sum() + b > 0 else -1

def f3(u_plus, u_minus, beta_plus, beta_minus, x_input):
    w = 2 * (beta_plus * u_plus - beta_minus * u_minus)
    b = - beta_plus * np.dot(u_plus ,u_plus) + beta_minus * np.dot(u_minus, u_minus)
    return 1 if (w * x_input).sum() + b > 0 else -1

def f1(u_plus, u_minus, beta_plus, beta_minus, x_input):
    w = 2 * (u_plus - u_minus)
    b = np.log(abs(beta_plus/beta_minus)) - np.dot(u_plus,u_plus) + np.dot(u_minus,u_minus)
    return 1 if (w * x_input).sum() + b > 0 else -1


def f(u_plus, u_minus, beta_plus, beta_minus, x_input):
    p1 = beta_plus * np.exp(-np.dot((x_input - u_plus),(x_input - u_plus)))
    p2 = beta_minus * np.exp(-np.dot((x_input - u_minus),(x_input - u_minus)))

    if p1 + p2 > 0:
        return 1
    else:
        return -1


def show_region(x1, x2,  beta1, beta2,fn):
    #[SHOW] Plot the decision boundary and training points
    start = -3
    end = 3
    npoints = 500
    xrange = np.linspace(start, end, npoints)
    yrange = np.linspace(start, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)

    X = np.array([xgrid.reshape(npoints * npoints), ygrid.reshape(npoints * npoints)]).T
    labels = [fn(x1, x2, beta1,beta2,x) for x in X]
    zgrid = np.array(labels).reshape(npoints, npoints)

    plt.figure()
    plt.title('RBF')
    plt.pcolor(xgrid, ygrid, zgrid)
    plt.plot(x1[0], x1[1],'ro')
    plt.plot(x2[0], x2[1],'go')
    plt.show()


if __name__ == '__main__':
    beta1 = 0.1
    beta2 = -1
    x1 = np.array([-1,0])
    x2 = np.array([1,1])
    show_region(x1, x2,  beta1, beta2,f4)
    show_region(x1, x2,  beta1, beta2,f)