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
    # since it's most likely been written in a very generalizable way.

    logger.warning("Problem 7.2 asks us to implement and run One-for-All,")
    logger.warning("but source code isn't actually provided to make that easy --")
    logger.warning("source code *is* however provided for the Multiclass Perceptron")
    logger.warning("which optimizes everything simultaneously.")
    logger.warning("")
    logger.warning("So what we're going to do is first run the code we're provided,")
    logger.warning("and /then/ run the 2-stage Softmax one by one as the problem")
    logger.warning("actually asks us to do.")

    def our_multiclass_perceptron(w):
        return multiclass_perceptron(w, x, y)

    # We want to initialize a weights matrix of the form (N+1) by C.
    # We know that in this case, C = 4; N+1, meanwhile, is however
    # many features the input data gives us.
    # So here, that should be 3x4.
    weight_matrix = (np.shape(x)[0] + 1, 4)

    logger.info("Generating random starting weights...")
    init_weights = (
        np.random.rand(weight_matrix[0], weight_matrix[1]).astype(np.float32) - 0.5
    )

    logger.info("Minimizing the Multiclass Perceptron cost function via GD...")
    (weights, costs) = gradient_descent(
        our_multiclass_perceptron, alpha=0.001, max_its=1000, w=init_weights
    )
    logger.info("Done!")
    logger.info("Final weights :: {}".format(weights[-1]))
    logger.info("Final cost    :: {}".format(costs[-1]))

    correct_count = 0
    misclassified_count = 0
    for i in range(0, x.shape[1]):
        print(y[:, i], np.argmax(model(x[:, i], weights[-1])))
        if y[:, i] == np.argmax(model(x[:, i], weights[-1])):
            correct_count += 1
        else:
            misclassified_count += 1
    print(correct_count, misclassified_count)

    logger.info("Okay, now let's minimize using the One-against-All approach.")
    logger.info("To do that, we just go through each classificiation and change")
    logger.info("the y_p values to -1, +1; then we minimize each weight as we did")
    logger.info("in the previous homeworks.")

    y_temp = np.zeros(y.shape)
    print(y_temp)
    for classification in [0, 1, 2, 3]:
        logger.info(
            "Now running One-for-All on classification: {}".format(classification)
        )
        for i in range(0, y.shape[1]):
            if int(y[0, i]) == classification:
                y_temp[0, i] = 1
            else:
                y_temp[0, i] = -1
        logger.debug("Softmax labels this run :: {}".format(y_temp))

        def our_softmax(w):
            return softmax(w, x, y_temp)

        (weights, costs) = gradient_descent(
            our_softmax,
            alpha=0.1,
            max_its=1000,
            w=np.random.rand(x.shape[0] + 1).astype(np.float32) - 0.5,
        )

        logger.debug("Final weight: {}".format(weights[-1]))
