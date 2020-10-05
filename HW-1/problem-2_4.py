#!/usr/bin/python3

# Andrew Quinn
# Dr. Katsaggelos, EECS 375 / 475
# Problem Set 1
# Problem 2.1

# See problem-2-1.png for the problem statement.

import math
import itertools
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")


def point_on_hypersphere(N=2, radius=1):
    """Returns a 1D array of a point on the N-sphere
    centered at the origin with a given radius.


    Credit to Wolfram MathWorld for the algorithm:

        https://mathworld.wolfram.com/HyperspherePointPicking.html

    """
    v = np.random.normal(size=N)
    return (v / np.linalg.norm(v)) * radius


def rosenbrock(w):
    assert len(w) == 2
    return 100 * (w[1] - w[0])**2 + (w[0] - 1)**2


def random_search_step_fixed(g, w_init, alpha=1, P=1000):
    dims = len(w_init)

    w_next = w_init + alpha * point_on_hypersphere(N=dims, radius=1)

    while P > 0:
        w_check = w_init + alpha * point_on_hypersphere(N=dims, radius=1)
        if g(w_check) < g(w_next):
            w_next = w_check
        P -= 1

    return (w_next, g(w_next))


def random_search_step_variable(g, w_init, alpha, current_step, P=1000):
    dims = len(w_init)

    w_next = w_init + alpha(current_step) * point_on_hypersphere(N=dims, radius=1)

    while P > 0:
        w_check = w_init + alpha(current_step) * point_on_hypersphere(N=dims, radius=1)
        if g(w_check) < g(w_next):
            w_next = w_check
        P -= 1

    return (w_next, g(w_next))


if __name__ == "__main__":

    # Change the seed to something other than 0 for actually new random
    # values. Otherwise you'll get the exact same values each time (which
    # is valuable for checking that you are getting the same data as me!).
    np.random.seed(0)

    print("Andrew Quinn\nEECS 375 - HW 1 - Problem 2.4\n" + ("-" * 80))

    global_minimum = rosenbrock(np.array([1, 1]))

    for i in range(1, 1000):
        not_necessarily_global_minimum = rosenbrock(np.array([1, 1])
                                                    + point_on_hypersphere())
        assert global_minimum <= not_necessarily_global_minimum
        # Program will crash if the above statement ceases to be true.

    K = 50
    w_0 = np.array([-2, -2])
    print((w_0, rosenbrock(w_0)))

    contour_coordinates = np.zeros((K, 2))
    contour_heights = np.zeros(K)

    contour_coordinates[0] = w_0
    contour_heights[0] = rosenbrock(w_0)

    for i in range(1, K):
        next_data = random_search_step_fixed(rosenbrock, w_0)
        contour_coordinates[i] = next_data[0]
        contour_heights[i] = next_data[1]
        assert contour_heights[i] == rosenbrock(next_data[0])
        w_0 = np.array(next_data[0])

    # I'd love to make a full on contour plot, but I just don't have the
    # time right now.

    plt.title("2.4 - fixed \\alpha - cost history function")
    plt.xlabel("Step K")
    plt.ylabel("rosenbrock(w)")
    plt.plot(list(range(1, 50+1)), contour_heights, '.')
    plt.show(block=False)
    plt.savefig(os.path.join(figs_dir, '2_4_fixed.png'))




    K = 50
    w_0 = np.array([-2, -2])
    print((w_0, rosenbrock(w_0)))

    contour_coordinates = np.zeros((K, 2))
    contour_heights = np.zeros(K)

    contour_coordinates[0] = w_0
    contour_heights[0] = rosenbrock(w_0)

    for i in range(1, K):
        next_data = random_search_step_variable(rosenbrock, w_0,
                                                alpha=lambda k: 1.0 / k,
                                                current_step=i)
        contour_coordinates[i] = next_data[0]
        contour_heights[i] = next_data[1]
        assert contour_heights[i] == rosenbrock(next_data[0])
        w_0 = np.array(next_data[0])

    # I'd love to make a full on contour plot, but I just don't have the
    # time right now.

    plt.title("2.4 - fixed vs. diminishing \\alpha - cost history function")
    plt.xlabel("Step K")
    plt.ylabel("rosenbrock(w)")
    plt.plot(list(range(1, 50+1)), contour_heights, '.')
    plt.show(block=False)
    plt.savefig(os.path.join(figs_dir, '2_4_fixed_vs_diminishing.png'))

    plt.clf()


    plt.title("2.4 - diminishing \\alpha = 1/k - cost history function")
    plt.xlabel("Step K")
    plt.ylabel("rosenbrock(w)")
    plt.plot(list(range(1, 50+1)), contour_heights, '.')
    plt.show(block=False)
    plt.savefig(os.path.join(figs_dir, '2_4_diminishing.png'))
