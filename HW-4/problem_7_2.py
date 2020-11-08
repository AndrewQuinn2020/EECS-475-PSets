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
four_class_data_url = "".join(
    [
        "https://raw.githubusercontent.com/",
        "jermwatt/machine_learning_refined/gh-pages/",
        "mlrefined_exercises/ed_2/mlrefined_datasets/",
        "superlearn_datasets/4class_data.csv",
    ]
)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

four_class_dataset_path = os.path.join(datasets_dir, "four_class_data.csv")


# Compute C linear combinations of the input points, one per classifier.
def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


# The convex softmax cost function
def softmax(w, x, y):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
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


if __name__ == "__main__":
    logger.info(
        "EECS 475 - Andrew Quinn - Problem 7.2 - Replicating a 4-Class Toy Model"
    )
    logger.info("-" * (88 - 11))
    logger.info("The Four Class CSV data we're training on can be found directly at:")
    logger.info("\t\t{}".format(four_class_data_url))

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(four_class_dataset_path):
        logger.warning("4-class toy dataset is missing... Downloading.")
        with open(four_class_dataset_path, "wb") as f:
            f.write(requests.get(four_class_data_url, allow_redirects=True).content)

    # Load in data.
    data = np.loadtxt(four_class_dataset_path, delimiter=",")

    # get input and output of dataset
    x = data[:-1, :]
    y = data[-1:, :]

    assert np.shape(x) == (2, 40)
    assert np.shape(y) == (1, 40)

    # We're going to attack this with the same gradient descent code we've been using,
    # since it's most likely been written in a very
