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
diagonal_stripes_data_url = "".join(
    [
        "https://raw.githubusercontent.com/jermwatt/machine_learning_refined/",
        "gh-pages/mlrefined_exercises/ed_2/mlrefined_datasets/",
        "nonlinear_superlearn_datasets/diagonal_stripes.csv",
    ]
)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

diagonal_stripes_dataset_path = os.path.join(datasets_dir, "diagonal_stripes.csv")


def model(x, theta):
    f = feature_transform(x, theta[0])
    a = theta[1][0] + np.dot(f.T, theta[1][1:])
    return a.T


# The Least squares cost function.
def least_squares(w, x, y):
    print(model(x, w).shape)
    print(model(x, w).reshape(1, -1).shape)
    print(y.shape)
    cost = np.sum((model(x, w) - y) ** 2)
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


def feature_transform(x, w):
    row = w[0] + np.sin(w[1] * (x[0, :] + x[1, :]) - w[2])
    return np.array([row, row])


if __name__ == "__main__":
    logger.info(
        "EECS 475 - Andrew Quinn - "
        "Problem 10.8 - Training a Diagonal Stripes differentiator"
    )
    logger.info("-" * (88 - 11))
    logger.info(
        "The transistor counts CSV data we're training on can be found directly at:"
    )
    logger.info("\t\t{}".format(diagonal_stripes_data_url))

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(diagonal_stripes_dataset_path):
        logger.warning("Transistor counts dataset is missing... Downloading.")
        with open(diagonal_stripes_dataset_path, "wb") as f:
            f.write(
                requests.get(diagonal_stripes_data_url, allow_redirects=True).content
            )

    data = np.loadtxt(diagonal_stripes_dataset_path, delimiter=",")

    x = data[:2, :]
    y = data[2:, :]

    assert x.shape == (2, 300)
    assert y.shape == (1, 300)

    # First let's test the feature transform code, to make sure it's giving us what we want.
    feature_test_array = np.array(
        [
            [0, 0, np.pi / 2, np.pi / 2, np.pi, np.pi],
            [0, np.pi / 2, np.pi / 2, 0, 0, np.pi],
        ]
    )
    logger.debug("Testing our feature transform.")
    logger.debug("Test array:\n{}".format(feature_test_array))
    logger.debug(
        "\n{}".format(
            np.round(feature_transform(feature_test_array, np.array([0, 1, 0])))
        )
    )
    logger.debug(
        "\n{}".format(
            np.round(feature_transform(feature_test_array, np.array([1, 1, 0])))
        )
    )
    logger.debug(
        "\n{}".format(
            np.round(feature_transform(feature_test_array, np.array([1, 1, np.pi / 2])))
        )
    )

    test_theta = np.array([[0, 1, 2], [3, 4, 5]])
    logger.debug("Testing our nonlinear model code.")
    logger.debug("test_theta = \n{}".format(test_theta))
    logger.debug("test_theta[0] = \n{}".format(test_theta[0]))
    logger.debug("test_theta[1][0] = \n{}".format(test_theta[1][0]))
    logger.debug("test_theta[1][1:] = \n{}".format(test_theta[1][1:]))
    logger.info(feature_transform(x, test_theta[0]))

    logger.info(model(x, test_theta))
    logger.info(least_squares(test_theta, x, y))

    def our_least_squares(w):
        return least_squares(w, x, y)

    weight_matrix = (np.shape(x)[0] + 1, 2)

    logger.info("Generating random starting weights...")
    init_weights = (
        np.random.rand(weight_matrix[0], weight_matrix[1]).astype(np.float32) - 0.5
    )

    logger.info("Minimizing the Least Squares cost function via GD...")
    (weights, costs) = gradient_descent(
        our_least_squares, alpha=0.001, max_its=10000, w=init_weights
    )
    logger.info("Done!")
    logger.info("Final weights :: {}".format(weights[-1]))
    logger.info("Final cost    :: {}".format(costs[-1]))
