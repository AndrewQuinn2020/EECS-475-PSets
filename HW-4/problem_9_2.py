#!/usr/bin/python3

# problem_7_2.py

import logging
import os
import pickle
import sys
from random import randint

import autograd.numpy as np
import colorlog
import requests
from autograd import grad, hessian
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)
# Change this to get more, or fewer, error messages.
#   DEBUG = Show me everything.
#   INFO = Only the green text and up.
#   WARNING = Only warnings.
#   ERROR = Only (user coded) error messages.
#   CRITICAL = Only (user coded) critical error messages.
logger.setLevel(colorlog.colorlog.logging.DEBUG)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

nmist_local_dataset_path = os.path.join(datasets_dir, "nmist_local_data.npz")


# Compute C linear combinations of the input points, one per classifier.
def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


# The convex softmax cost function
def softmax(w, x, y):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
    return cost / float(np.size(y))


def multiclass_perceptron(w, x, y, lam=10 ** -5):
    all_evals = model(x, w)
    a = np.max(all_evals, axis=0)
    b = all_evals[y.astype(int).flatten(), np.arange(np.size(y))]
    cost = np.sum(a - b)
    cost += lam * np.linalg.norm(w[1:, :], "fro") ** 2
    return cost / float(np.size(y))


# We are limited to using zero- or first-order techniques for this one, which means
# gradient descent is the name of the game today.
def gradient_descent(g, alpha, max_its, w):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]  # container for weight history
    cost_history = [g(w)]  # container for corresponding cost function history
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha * grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history, cost_history


def newtons_method(g, max_its, w, epsilon=(10 ** (-7)), verbose=False):
    # Compute gradiant and Hessian using autograd.
    gradient = grad(g)
    hess = hessian(g)

    # Run Newton's method loop.
    weight_history = [w]
    cost_history = [g(w)]

    for k in range(0, max_its):
        grad_eval = gradient(w)
        hess_eval = hess(w)

        # Reshape Hessian to be square matrix.
        hess_eval.shape = (
            int((np.size(hess_eval)) ** (0.5)),
            int((np.size(hess_eval)) ** (0.5)),
        )

        # Solve second-order system for weight update.
        A = hess_eval + epsilon * np.eye(w.size)
        b = grad_eval
        w = np.linalg.solve(A, np.dot(A, w) - b)

        # Record weight and cost.
        weight_history.append(w)
        cost_history.append(g(w))
    return (weight_history, cost_history)


if __name__ == "__main__":
    logger.info("EECS 475 - Andrew Quinn - Problem 9.2 - MNIST training")
    logger.info("-" * (88 - 11))

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(nmist_local_dataset_path):
        logger.warning("NMIST 784 data isn't locally pickled! This might take a bit.")
        # import MNIST
        logger.info("Fetching NMIST 784...")
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        logger.info("... Fetch completed!")
        # re-shape input/output data
        x = x.T
        y = np.array([int(v) for v in y])[np.newaxis, :]

        assert np.shape(x) == (784, 70000)
        assert np.shape(y) == (1, 70000)

        logger.info("Saving locally to {}".format(nmist_local_dataset_path))
        np.savez(nmist_local_dataset_path, x, y)
        logger.info("Save completed!")
        logger.info("You should now be able to load them back using\n")
        logger.info("    numpy.load('{}')\n".format(nmist_local_dataset_path))
    else:
        logger.warning("Loading data locally from {}".format(nmist_local_dataset_path))
        npzfile = np.load(nmist_local_dataset_path)
        logger.info("Files in the local archive: {}".format(npzfile.files))
        x = npzfile["arr_0"]
        y = npzfile["arr_1"]

        assert np.shape(x) == (784, 70000)
        assert np.shape(y) == (1, 70000)

    logger.info("")
