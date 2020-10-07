#!/usr/bin/python3

# Andrew Quinn
# Dr. Katsaggelos, EECS 375 / 475
# Problem Set 1
# Problem 2.1 d

import os
import sys

import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")


def matrix_multiplication_test():
    """Just a bunch of code to help me remember the basics."""
    w = np.array([[2],[3]])
    print(w)
    print(np.transpose(b))
    print(np.matmul(np.transpose(b), w))
    print(np.matmul(np.transpose(w), b))
    print(np.matmul(w, np.transpose(b)))
    print(np.matmul(b, np.transpose(w)))

    print(np.matmul(np.matmul(np.transpose(w), C), w))
    print(np.matmul(np.transpose(b), w) + np.matmul(np.matmul(np.transpose(w), C), w))
    print(1000 + np.matmul(np.transpose(b), w) + np.matmul(np.matmul(np.transpose(w), C), w))

    return None


def quadratic(w, a, b, C):
    """Multi-dimensional quadratic equation of the form

        g(w) = a + (b^T) w + (w^T) C w

    where a is a scalar, w, b are 1d vectors of length n,
    and C an n*n symmetric matrix."""

    return (a + np.matmul(np.transpose(b), w) + np.matmul(np.matmul(np.transpose(w), C), w))[0,0]


if __name__ == "__main__":
    print("Andrew Quinn - EECS 375/475 - Plotting Problem 3.1.(d)")
    print("-" * 80)

    a = 0
    b = np.array([[1],[1]])
    C = np.array([[2,1],[1,3]])

    def g(w): return quadratic(w, a, b, C)
    def g_pos(w1, w2): return g([[w1],[w2]])
    # First let's test to make sure our terms actually make sense here.
    # matrix_multiplication_test()

    # Now let's generate the actual grid we want.
    # By our calculations, the point we're looking for is at
    #
    #    w = [(-2/5),(-1/5)]
    #
    # so we just want to plot the neighborhood around that point.

    nbhd_dist = 0.2
    res = 0.1
    w_crit = np.array([[-2/5],[-1/5]])

    print(g(w_crit))

    # print(w_crit[0,0])
    # print(w_crit[1,0])

    print(g_pos(-2/5, -1/5))

    d = 1.0
    res = 0.01
    center_x = w_crit[0,0]
    center_y = w_crit[1,0]

    x = np.arange(center_x - d, center_x + d + res, res)
    y = np.arange(center_y - d, center_y + d + res, res)
    print(x)
    print(y)
    X,Y = np.meshgrid(x, y) # grid of point
    Z = np.vectorize(g_pos)(X, Y) # evaluation of the function on the grid

    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.show()
