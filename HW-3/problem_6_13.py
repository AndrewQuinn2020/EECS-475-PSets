#!/usr/bin/python3

# problem_6_13.py

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
breast_cancer_data_url = "".join(
    [
        "https://raw.githubusercontent.com/",
        "jermwatt/machine_learning_refined/gh-pages/",
        "mlrefined_exercises/ed_2/mlrefined_datasets/",
        "superlearn_datasets/breast_cancer_data.csv",
    ]
)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

breast_cancer_dataset_path = os.path.join(datasets_dir, "breast_cancer_data.csv")


# compute linear combination of input point
def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


# the convex softmax cost function
def softmax(w, x, y):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
    return cost / float(np.size(y))


# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
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
    if not verbose:
        logger.disabled = True
    """The Newton's Method code from _Machine Learning Refined_."""

    # Compute gradiant and Hessian using autograd.
    logger.debug("Constructing gradient and Hessian.")
    gradient = grad(g)
    logger.debug("Gradient constructed.")
    hess = hessian(g)
    logger.debug("Hessian constructed.")

    # Run Newton's method loop.
    logger.debug("Our first weight is {}.".format(w))
    weight_history = [w]
    logger.debug("Our first cost is {}.".format(g(w)))
    cost_history = [g(w)]

    for k in range(0, max_its):
        logger.debug("Iteration {} ::".format(k))

        grad_eval = gradient(w)
        hess_eval = hess(w)

        logger.debug("\tgradient @ {} = {}".format(w, grad_eval))

        # Reshape Hessian to be square matrix.
        hess_eval.shape = (
            int((np.size(hess_eval)) ** (0.5)),
            int((np.size(hess_eval)) ** (0.5)),
        )

        logger.debug("\tHessian @ {} = \n{}".format(w, hess_eval))

        # Solve second-order system for weight update.
        A = hess_eval + epsilon * np.eye(w.size)
        b = grad_eval
        w = np.linalg.solve(A, np.dot(A, w) - b)

        logger.debug("\tNew w @ {},".format(w))
        logger.debug("\twith cost g(w) = {}.".format(g(w)))

        # Record weight and cost.
        weight_history.append(w)
        cost_history.append(g(w))

    if not verbose:
        logger.disabled = False

    return (weight_history, cost_history)


def pickle_costs_and_weights(costs, weights, pname, verbose=False):
    if not verbose:
        logger.disabled = True
    cost_loc = os.path.join(pickle_dir, "{}_cost_history.pickle".format(pname))
    weight_loc = os.path.join(pickle_dir, "{}_weight_history.pickle".format(pname))

    with open(cost_loc, "wb") as fp:
        pickle.dump(costs, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("cost_history pickled to {}".format(fp.name))
        logger.info("Load it back in with")
        logger.info('\t `cost_history = pickle.load(open("{}", "rb"))`'.format(fp.name))

    with open(weight_loc, "wb") as fp:
        pickle.dump(weights, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("weight_history pickled to {}".format(fp.name))
        logger.info("Load it back in with")
        logger.info('\t `cost_history = pickle.load(open("{}", "rb"))`'.format(fp.name))

    if not verbose:
        logger.disabled = False

    return None


def spot_check_trained_model(x, y, w, n, verbose=True):
    """Debug version of check_classify. Prints helpful statements if verbose=True."""
    if not verbose:
        logger.disabled = True

    classification = model(x[:, n], w)
    logger.debug("model(x[:, {}], weights) returns {:>40}.".format(n, classification))
    logger.debug("The true classification was {:>40}".format(str(y[:, n][0])))
    if not verbose:
        logger.disabled = False

    return np.sign(classification) == np.sign(y[:, n][0])


def check_classify(x, y, w, n):
    """Returns True if the model classified the n'th input data correctly."""
    return np.sign(model(x[:, n], w)) == np.sign(y[:, n][0])


if __name__ == "__main__":
    logger.info("EECS 475 - Andrew Quinn - Problem 6.13 - Softmax vs Perceptron")
    logger.info("-" * (88 - 11))
    logger.info("The Boston CSV data we're training on can be found directly at:")
    logger.info("\t\t{}".format(breast_cancer_data_url))

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(breast_cancer_dataset_path):
        logger.warning("Breast cancer dataset is missing... Downloading.")
        with open(breast_cancer_dataset_path, "wb") as f:
            f.write(requests.get(breast_cancer_data_url, allow_redirects=True).content)

    # Load in data.
    data = np.loadtxt(breast_cancer_dataset_path, delimiter=",")

    # get input and output of dataset
    x = data[:-1, :]
    y = data[-1:, :]

    logger.debug("Shape of inputs (should be (8, 699):  {}".format(np.shape(x)))
    logger.debug("Shape of ourputs (should be (1, 699): {}".format(np.shape(y)))

    if np.shape(x) != (8, 699):
        logger.warning("The data for breast_cancer.csv isn't the expected shape.")
    if np.shape(y) != (1, 699):
        logger.warning("The data for breast_cancer.csv isn't the expected shape.")

    logger.info("Since Softmax is provably everywhere convex, we can use")
    logger.info("Newton's method to get the definition *really* fast so long")
    logger.info("as our dataset is of a low-enough dimensionality.")

    logger.debug("Loading Softmax with our input/output pairs...")

    def our_softmax(w):
        return softmax(w, x, y)

    logger.debug("Running Newton's method on our Softmax...")
    (weights, costs) = newtons_method(
        our_softmax, 10, np.zeros(9).astype(np.float32), verbose=True
    )

    pickle_costs_and_weights(costs, weights, "6_13_softmax", verbose=True)

    logger.info("Softmax completed!")

    spots = 100
    while spots > 0:
        logger.debug(
            spot_check_trained_model(x, y, weights[-1], randint(0, x.shape[1] - 1))
        )
        spots -= 1

    logger.info("Counting misclassifications in Softmax model...")

    misclassifications = 0
    for i in range(0, x.shape[1]):
        if not check_classify(x, y, weights[-1], i):
            logger.debug("Input {} was misclassified.".format(i))
            misclassifications += 1

    logger.info("{} misses under Newton's Method/Softmax.".format(misclassifications))

    logger.debug("Running gradient descent on our Softmax...")

    (weights, costs) = gradient_descent(
        our_softmax, 0.01, 10000, np.zeros(9).astype(np.float32)
    )

    pickle_costs_and_weights(costs, weights, "6_13_softmax", verbose=True)

    logger.info("Softmax completed!")

    spots = 100
    while spots > 0:
        logger.debug(
            spot_check_trained_model(x, y, weights[-1], randint(0, x.shape[1] - 1))
        )
        spots -= 1

    logger.info("Counting misclassifications in Softmax model...")

    misclassifications = 0
    for i in range(0, x.shape[1]):
        if not check_classify(x, y, weights[-1], i):
            logger.debug("Input {} was misclassified.".format(i))
            misclassifications += 1

    logger.info("{} misses under gradient descent/Softmax.".format(misclassifications))
