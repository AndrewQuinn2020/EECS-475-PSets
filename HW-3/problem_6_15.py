#!/usr/bin/python3

# problem_6_15.py

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
logger.setLevel(colorlog.colorlog.logging.INFO)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# I had issues with when I wanted to clear out directories and run the code fresh, so
# we're going to use a lil `requests` magic to pull the CSV we want directly from the
# Github repo.
credit_dataset_url = "".join(
    [
        "https://raw.githubusercontent.com/",
        "jermwatt/machine_learning_refined/gh-pages/",
        "mlrefined_exercises/ed_2/mlrefined_datasets/",
        "superlearn_datasets/credit_dataset.csv",
    ]
)

script_dir = os.path.dirname(__file__)
figs_dir = os.path.join(script_dir, "figs")
pickle_dir = os.path.join(script_dir, "pickles")
datasets_dir = os.path.join(script_dir, "datasets")

credit_dataset_name = "credit_dataset.csv"
credit_dataset_path = os.path.join(datasets_dir, credit_dataset_name)


# Given a url and a path, download data to that path if there isn't something there.
def download_data_if_needed(data_url, data_path):
    data_name = os.path.splitext(data_path)[0]

    for dir in [script_dir, figs_dir, pickle_dir, datasets_dir]:
        if not os.path.exists(dir):
            logger.warning(
                "Directory \t\t{}\t\t doesn't exist... Creating.".format(dir)
            )
            os.makedirs(dir)

    if not os.path.exists(data_path):
        logger.warning("Dataset is missing... Downloading to {}".format(data_name))
        with open(data_path, "wb") as f:
            f.write(requests.get(data_url, allow_redirects=True).content)


# compute linear combination of input point
def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


# the convex softmax cost function
def softmax(w, x, y):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
    return cost / float(np.size(y))


# The perceptron direct cost function
def perceptron(w, x, y):
    cost = np.sum(np.maximum(0, -y * model(x, w)))
    return cost / float(np.size(y))


# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g, alpha, max_its, w):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]  # container for weight history
    cost_history = [g(w)]  # container for corresponding cost function history
    logger.warning(g(w))
    for k in range(max_its):
        logger.warning("Loop {}".format(k))
        # evaluate the gradient, store current weights and cost function value
        logger.warning(gradient(w))
        grad_eval = gradient(w)
        logger.warning(grad_eval)

        # take gradient descent step
        w = w - alpha * grad_eval
        logger.warning(w)

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


def confusion_matrix(x, y, weights, verbose=False):
    """Returns a 2x2 NumPy matrix of integers corresponding to

                actual y_p
        +1                    -1
    [[True_Postiives    False_Negatives]        +1    expected y_p
     [False_Positives   True_Negatives ]]       -1"""

    if not verbose:
        logger.disabled = True

    confusion_matrix = np.array([[0, 0], [0, 0]])
    logger.debug("Initializing confusion_matrix: \n{}".format(confusion_matrix))

    for i in range(0, x.shape[1]):
        y_pred = model(x[:, i], w)
        y_true = y[:, i][0]

        if np.sign(model(x[:, i], w)) == np.sign(y[:, i][0]):
            verdict = "True_"
            if np.sign(model(x[:, i], w)) > 0:
                verdict += "Positive"
                confusion_matrix[0, 0] += 1
            else:
                verdict += "Negative"
                confusion_matrix[1, 1] += 1
        else:
            verdict = "False_"
            if np.sign(model(x[:, i], w)) > 0:
                verdict += "Positive"
                confusion_matrix[1, 0] += 1
            else:
                verdict += "Negative"
                confusion_matrix[0, 1] += 1

        logger.debug(
            "predicted {:>10.4f}     actual {:>10.4f}    verdict {:>20}".format(
                y_pred, y_true, verdict
            )
        )

    logger.info("Final confusion_matrix: \n{}".format(confusion_matrix))

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
    logger.info("EECS 475 - Andrew Quinn - Problem 6.15 - Credit check")
    logger.info("-" * (88 - 11))
    logger.info("The credit CSV data we're training on can be found directly at:")
    logger.info("\t\t{}".format(credit_dataset_url))

    download_data_if_needed(credit_dataset_url, credit_dataset_path)
    # Load in data.
    data = np.loadtxt(credit_dataset_path, delimiter=",")

    # get input and output of dataset

    # Get x, but replace any nans with 0s and any infs with the largest floats we can.
    x = np.nan_to_num(data[:-1, :])
    # This is called a "broadcast operation" if you want to look it up.
    # Calculate the mean of each row of x, and subtract from the row elementwise.
    x = x - x.mean(axis=1)[:, np.newaxis]
    # (Now, each row should have a mean very close to 0.)
    # Calculate the standard deviation of each row of x, and divide the row elementwise.
    x = x / x.std(axis=1)[:, np.newaxis]
    # (Now, x has been standard-normalized, and you should be good to go.)
    y = data[-1:, :]

    logger.debug("Shape of inputs (should be (20, 1000):  {}".format(np.shape(x)))
    logger.debug("Shape of ourputs (should be (1, 1000):  {}".format(np.shape(y)))

    if np.shape(x) != (20, 1000):
        logger.warning(
            "Input data for {} isn't the expected shape.".format(credit_dataset_name)
        )
    if np.shape(y) != (1, 1000):
        logger.warning(
            "Output data for {} isn't the expected shape.".format(credit_dataset_name)
        )

    def our_softmax(w):
        return softmax(w, x, y)

    logger.debug("Running Newton's method on our Softmax...")
    (weights, costs) = newtons_method(
        our_softmax,
        max_its=10,
        w=np.zeros(x.shape[0] + 1).astype(np.float32),
    )
    w = weights[-1]

    logger.info("Generating confusion matrix for Softmax/Newton's Method...")
    confusion_matrix(x, y, w, verbose=True)
    logger.info("Softmax/Newton's Method complete!")

    def our_perceptron(w):
        return softmax(w, x, y)

    logger.debug("Running Newton's method on our Perceptron...")
    (weights, costs) = newtons_method(
        our_perceptron,
        max_its=10,
        w=np.zeros(x.shape[0] + 1).astype(np.float32),
    )
    w = weights[-1]

    logger.info("Generating confusion matrix for Perceptron/Newton's Method...")
    confusion_matrix(x, y, w, verbose=True)
    logger.info("Perceptron/Newton's Method complete!")

    logger.warning("Beginning to try out gradient descent.")
    for alpha in [10 ** -1, 10 ** -2, 10 ** -3]:
        logger.debug("Running GD (alpha = {}) on our Softmax...".format(alpha))
        (weights, costs) = gradient_descent(
            our_softmax,
            alpha=alpha,
            max_its=10,
            w=np.zeros(x.shape[0] + 1).astype(np.float64),
        )
        w = weights[-1]

        logger.info("Generating confusion matrix for Softmax/Newton's Method...")
        confusion_matrix(x, y, w, verbose=True)
        logger.info("GD (alpha = {})/Newton's Method complete!".format(alpha))
