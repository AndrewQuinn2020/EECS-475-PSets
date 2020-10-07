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

w_crit = np.array([[-2/5],[-1/5]])
C = np.array([[2,1],
              [1,3]])
b = np.array([[1],[1]])

def g(w):
    term_1 = 0.5 * np.matmul(np.matmul(np.transpose(w), C), w)
    term_2 = np.matmul(np.transpose(b), w)
    return (term_1 + term_2)[0,0]

def g_2d(x, y):
    return g(np.array([[x],[y]]))

if __name__ == "__main__":
    print("Andrew Quinn - EECS 375/475 - Plotting Problem 3.1.(d)")
    print("-" * 80)

    print(g(w_crit))
    print(g_2d(-2/5, -1/5))
