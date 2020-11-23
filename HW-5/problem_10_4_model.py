#!/usr/bin/python3

# problem_7_2.py

import logging
import os
import sys
from random import randint

import pandas as pd
import autograd.numpy as np
import colorlog
import requests
from autograd import grad, hessian
from matplotlib import pyplot as plt

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

# I had issues with when I wanted to clear out directories and run the code fresh, so
# we're going to use a lil `requests` magic to pull the CSV we want directly from the
# Github repo.
transistor_counts_data_url = "".join(
    [
        "https://raw.githubusercontent.com/jermwatt/machine_learning_refined/",
        "gh-pages/mlrefined_exercises/ed_2/mlrefined_datasets/",
        "nonlinear_superlearn_datasets/transistor_counts.csv",
    ]
)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

transistor_counts_dataset_path = os.path.join(datasets_dir, "transistor_counts.csv")


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


def linearize(y):
    return np.log2(y)


def delinearize(y):
    return y ** 2


if __name__ == "__main__":
    logger.info(
        "EECS 475 - Andrew Quinn - Problem 7.2 - Replicating a 4-Class Toy Model"
    )
    logger.info("-" * (88 - 11))
    logger.info(
        "The transistor counts CSV data we're training on can be found directly at:"
    )
    logger.info("\t\t{}".format(transistor_counts_data_url))

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(transistor_counts_dataset_path):
        logger.warning("Transistor counts dataset is missing... Downloading.")
        with open(transistor_counts_dataset_path, "wb") as f:
            f.write(
                requests.get(transistor_counts_data_url, allow_redirects=True).content
            )

    # Load in data.
    data = np.asarray(pd.read_csv(transistor_counts_dataset_path, header=None))
    x = data[:, 0]
    x.shape = (len(x), 1)
    y = data[:, 1]
    y.shape = (len(y), 1)

    assert np.shape(x) == (85, 1)
    assert np.shape(y) == (85, 1)

    # We need to transform y into a logarithm, train a linear model on that, and then
    # run any points we predict through an inverse transform to get actual data we can
    # use.

    # Let's actually make two graphs first, to confirm that we have the scatter looking correct.

    # This code reproduces the original graph.
    plt.scatter(x, y)
    plt.title("Problem 10.4 - Original graph reproduction")
    plt.ylabel("Year")
    plt.xlabel("Transistor count")
    plt.draw()
    plt.savefig(os.path.join(figs_dir, "problem_10_4_original.png"))
    plt.close()
    logger.warning("Original graph reproducted.")

    # This code linearizes y so we can spot-check it.
    plt.scatter(x, linearize(y))
    plt.title("Problem 10.4 - Original data, run through log_2")
    plt.ylabel("Year")
    plt.xlabel("ln_2(Transistor count)")
    plt.draw()
    plt.savefig(os.path.join(figs_dir, "problem_10_4_output_linearized.png"))
    plt.close()
    logger.warning("Output-linearized graph reproducted.")

    plt.scatter(x - x[0], linearize(y))
    plt.title("Problem 10.4 - Original data, run through log_2, years normalized")
    plt.ylabel("(Year - Start Year)")
    plt.xlabel("ln_2(Transistor count)")
    plt.draw()
    plt.savefig(
        os.path.join(figs_dir, "problem_10_4_output_linearized_year_pulled_back.png")
    )
    plt.close()
    logger.warning("Output-linearized, year-corrected graph reproducted.")
