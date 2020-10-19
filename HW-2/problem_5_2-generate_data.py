#!/usr/bin/python3

import logging
import os
import pickle
import sys
from random import randint

import requests
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
logger.setLevel(colorlog.colorlog.logging.INFO)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# I had issues with when I wanted to clear out directories and run the code fresh, so
# we're going to use a lil `requests` magic to pull the CSV we want directly from the
# Github repo.
kleiber_data_url = "".join(
    [
        "https://raw.githubusercontent.com/",
        "jermwatt/machine_learning_refined/gh-pages/",
        "mlrefined_exercises/ed_2/mlrefined_datasets/",
        "superlearn_datasets/kleibers_law_data.csv",
    ]
)


script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

kleiber_dataset_path = os.path.join(datasets_dir, "kleiber.csv")


def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


def least_squares(w, x, y):
    cost = np.sum((model(x, w) - y) ** 2)
    return cost / float(np.size(y))


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
    logger.info("The Kleiber CSV data we're training on can be found directly at:")
    logger.info("\t\t{}".format(kleiber_data_url))

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(kleiber_dataset_path):
        logger.warning("Kleiber dataset is missing... Downloading.")
        with open(kleiber_dataset_path, "wb") as f:
            f.write(requests.get(kleiber_data_url, allow_redirects=True).content)

    data = np.loadtxt(kleiber_dataset_path, delimiter=",")
    x = np.log(data[:-1, :])
    y = np.log(data[-1:, :])

    # Look over to make sure this looks roughly linear.
    # plt.scatter(x, y)
    # plt.title("Does this look roughly linear to you?")
    # plt.show()
    # plt.close()

    def our_least_squares(w):
        return least_squares(w, x, y)

    (weights, costs) = newtons_method(
        our_least_squares, 10, np.array([0, 0]).astype(np.float32)
    )

    logger.info("Least squares complete!")
    m, b = np.polyfit(x[0, :], y[0, :], 1)
    logger.debug("Polyfit numpy :: y = {:.4f} + {:.4f} x".format(b, m))
    logger.info(
        "Final weights  :: y = {:.4f} + {:.4f} x".format(weights[-1][0], weights[-1][1])
    )
    logger.debug("If polyfit doesn't match our model, something went wrong.")
    logger.info("Final costs   :: {:.4f}".format(costs[-1]))

    y_fit = [weights[-1][1] * xi + weights[-1][0] for xi in x]

    plt.scatter(x, y)
    plt.scatter(x, y_fit, linewidth=2)
    plt.title("$\ln{y_p}$ vs $\ln{x_p}$")
    plt.ylabel("$\ln{y_p}$")
    plt.xlabel("$\ln{x_p}$")
    plt.show(block=False)

    pickle_costs_and_weights(costs, weights, "5_2")
