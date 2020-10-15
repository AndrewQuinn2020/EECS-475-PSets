#!/usr/bin/python3

# Andrew Quinn
# Dr. Katsaggelos, EECS 375 / 475
# Problem Set 1
# Problem 2.1 d

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")

w_crit = np.array([[-2 / 5], [-1 / 5]])
C = np.array([[2, 1], [1, 3]])
b = np.array([[1], [1]])


def g(w):
    term_1 = 0.5 * np.matmul(np.matmul(np.transpose(w), C), w)
    term_2 = np.matmul(np.transpose(b), w)
    return (term_1 + term_2)[0, 0]


def g_2d(x, y):
    return g(np.array([[x], [y]]))


if __name__ == "__main__":
    print("Andrew Quinn - EECS 375/475 - Plotting Problem 3.1.(d)")
    print("-" * 80)

    print(g(w_crit))
    print(g_2d(-2 / 5, -1 / 5))
    print(g_2d(0, 0))

    d = 1
    res = 0.01
    c_x = w_crit[0, 0]
    c_y = w_crit[1, 0]
    # print(c_x, c_y)

    x = np.arange(c_x - d, c_x + d + res, res)
    y = np.arange(c_y - d, c_y + d + res, res)
    # print(x)
    # print(y)

    xx, yy = np.meshgrid(x, y)
    # print(xx)
    # print(yy)
    #
    # print(np.vectorize(g_2d)(xx, yy))
    # Yes! Finally!

    zz = np.vectorize(g_2d)(xx, yy)

    plt.contourf(xx, yy, zz, cmap=cm.coolwarm)
    plt.colorbar()
    plt.title("g(w) - 2D contour")
    plt.show(block=False)
    plt.savefig(os.path.join(figs_dir, "3_1_d_2d_contour.png"))
    plt.close

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cm.coolwarm,
        linewidth=1,
        antialiased=False,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )

    cset = ax.contourf(xx, yy, zz, zdir="z", offset=-100, cmap=cm.coolwarm)
    cset = ax.contourf(xx, yy, zz, zdir="x", offset=-40, cmap=cm.coolwarm)
    cset = ax.contourf(xx, yy, zz, zdir="y", offset=40, cmap=cm.coolwarm)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title("g(w) - 3D contour")
    plt.show(block=False)
    plt.savefig(os.path.join(figs_dir, "3_1_d_3d_contour.png"))
    plt.close()
