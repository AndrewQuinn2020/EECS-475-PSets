#!/usr/bin/python3

# Andrew Quinn
# Dr. Katsaggelos, EECS 375 / 475
# Problem Set 1
# Problem 2.1 d

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from jax import grad

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")


def g(w):
    return (1/50) * (w**4 + w**2 + 10*w)

def g_prime(w):
    return (1/25) * (2*(w**3) + w + 5)

def descend(g, g_prime, alpha, max_iters, w):
    weights = [w]
    costs = [g(w)]

    for iter in range(0, max_iters):
        w -= alpha * g_prime(w)
        weights.append(w)
        costs.append(g(w))

    return np.array([weights, costs])


def plot_descent(descent, save_loc=None, show=False, block=True):
    fig, axs = plt.subplots(2)

    fig.suptitle("Descent of g(w) - Weight and cost histories")

    axs[0].set_title("Weight history")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("w")
    axs[0].plot(descent[0,:])
    axs[1].set_title("Cost history")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("g(w)")
    axs[1].plot(descent[1,:], 'tab:orange')

    # Need this so that labels don't overlap.
    plt.tight_layout()

    if save_loc != None:
        plt.savefig(os.path.join(figs_dir, save_loc))

    if show:
        plt.show(block=block)
        plt.close()

    return fig, axs


if __name__ == "__main__":
    print("Andrew Quinn - EECS 375/475 - Gradient descents for 3.5")
    print("-" * 80)
    print(g(0))
    print(g(1))

    w_0 = 2
    max_its = 1000
    alphas = list(map(lambda s: 10**(-s), range(0, 3)))
    print(alphas)

    c = 1
    for alpha in alphas:
        descent = descend(g, g_prime, alpha, max_its, w_0)
        plot_descent(descent, "3_5_descent_alpha_{}.png".format(c))
        c += 1
