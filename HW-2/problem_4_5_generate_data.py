#!/usr/bin/python3

import logging
import os
import pickle
import sys

import autograd.numpy as np
import colorlog
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
np.set_printoptions(linewidth=10000)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")


for dir in [script_dir, figs_dir, pickle_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)


def newtons_method(g, max_its, w, epsilon=(10 ** (-7))):
    """The Newton's Method code from _Machine Learning Refined_."""

    # Compute gradiant and Hessian using autograd.
    logger.debug("Constructing gradient and Hessian.")
    gradient = grad(g)
    logger.debug("Gradient constructed.")
    hess = hessian(g)
    logger.debug("Hessian constructed.")

    # Run Newton's method loop.
    weight_history = [w]
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

        logger.debug("\tNew w @ {}, with new cost g(w) = {}".format(w, g(w)))

        # Record weight and cost.
        weight_history.append(w)
        cost_history.append(g(w))

    return (weight_history, cost_history)


def g_test(w):
    """An easier function to debug the raw numerical output of."""

    return w[0] ** 2 + w[1] ** 2


def g(w):
    """The actual function we're working with."""

    return np.log(1.0 + np.exp(w[0] ** 2 + w[1] ** 2))


def pickle_costs_and_weights(costs, weights, pname):
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

    return None


if __name__ == "__main__":
    logger.info("EECS 475 - Andrew Quinn - Problem 4.5.(c) and (d) - data gen")
    logger.info("-" * (88 - 11))
    logger.debug(np.array([[1, 1, 1]]))

    if not np.isclose(np.log(2), g(np.array([0, 0]))):
        logger.warning("Something seems wrong with g(w). g(0) != ln 2 ~= 0.69315.")

    # Code to generate 4.5.(c) data.
    start = np.array([1, 1]).astype(np.float32)
    logger.debug("Start point: {}, with g({}) = {}".format(start, start, g(start)))

    cost_history, weight_history = newtons_method(g, 10, start)

    logger.debug(cost_history)
    logger.debug(weight_history)

    pickle_costs_and_weights(cost_history, weight_history, "4_5_c")

    # Code to generate 4.5.(d) data.
    start = np.array([4, 4]).astype(np.float32)
    logger.debug("Start point: {}, with g({}) = {}".format(start, start, g(start)))

    cost_history, weight_history = newtons_method(g, 10, start)

    logger.debug(cost_history)
    logger.debug(weight_history)

    pickle_costs_and_weights(cost_history, weight_history, "4_5_d")
