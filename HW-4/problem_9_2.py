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
from skimage import feature

# This problem in particular can take a LONG time to work. Unless you're *absolutely
# sure* your code is working, I would leave this number very low.
ITERATIONS = 100

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
nmist_local_canny_dataset_path = os.path.join(
    datasets_dir, "nmist_local_data_canny.npz"
)


# Given a set of input features x, fill in any `nan` values with the mean of the row.
def nans_to_row_mean(inputs):
    pass


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
def gradient_descent(g, alpha, max_its, w, x, y):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]  # container for weight history
    cost_history = [g(w)]  # container for corresponding cost function history
    misses_history = [
        check_classify(w, x, y)[1]
    ]  # container for how many points get *mis*classified over time.

    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha * grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
        misses_history.append(check_classify(w, x, y)[1])

    return weight_history, cost_history, misses_history


def check_classify(w, x, y):
    """Given a set of weights, a set of input features, and a set of output features, return a 2-tuple of how
    many inputs get correctly classified."""
    correct_count = 0
    misclassified_count = 0
    for i in range(0, x.shape[1]):
        if y[:, i] == np.argmax(model(x[:, i], w)):
            correct_count += 1
        else:
            misclassified_count += 1
    return (correct_count, misclassified_count)


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
        logger.warning("NMIST 784 data isn't locally archived! This might take a bit.")
        # import MNIST
        logger.info("Fetching NMIST 784...")
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        logger.info("... Fetch completed!")
        # re-shape input/output data
        x = x.T
        y = np.array([int(v) for v in y])[np.newaxis, :]
        logger.info("Saving locally to {}".format(nmist_local_dataset_path))
        np.savez(nmist_local_dataset_path, x, y)
        logger.info("Save completed!")
        logger.info("You should now be able to load them back anytime using\n")
        logger.info("    numpy.load('{}')\n".format(nmist_local_dataset_path))
        logger.warning(
            "We will load locally from here on out unless you delete the file."
        )
    else:
        logger.warning("Loading local data from {}...".format(nmist_local_dataset_path))
        npzfile = np.load(nmist_local_dataset_path)
        x = npzfile["arr_0"]
        y = npzfile["arr_1"]
        logger.info("Local data loaded!")

    assert np.shape(x) == (784, 70000)
    assert np.shape(y) == (1, 70000)

    logger.debug("What we want to do here is compare a 'traditional' Softmax descent")
    logger.debug("to a Softmax descent based on the edge histogram data.")

    # We're going to attack this with the same gradient descent code we've been using,
    # since it's most likely been written in a very generalizable way.

    def our_multiclass_perceptron(w):
        return multiclass_perceptron(w, x, y)

    # We want to initialize a weights matrix of the form (N+1) by C. We know that in
    # this case, C = 10, because there are 10 digits; N+1, meanwhile, is however many
    # features the input data gives us. So here, that should be 785 x 10.
    weight_matrix = (np.shape(x)[0] + 1, 10)

    logger.info("Generating random starting weights...")
    init_weights = (
        np.random.rand(weight_matrix[0], weight_matrix[1]).astype(np.float32) - 0.5
    )

    logger.info("Minimizing the Multiclass Perceptron cost function via GD...")
    (weights, costs, misses) = gradient_descent(
        our_multiclass_perceptron,
        alpha=0.01,
        max_its=ITERATIONS,
        w=init_weights,
        x=x,
        y=y,
    )
    logger.info("Done!")
    logger.info("Final cost    :: {}".format(costs[-1]))
    logger.info("Final misses  :: {}".format(misses[-1]))

    original_weights = weights
    original_costs = costs
    original_misses = misses

    (correct_count, misclassified_count) = check_classify(weights[-1], x, y)
    logger.warning(
        "MULTICLASS PERCEPTRON RESULTS: {}, {}".format(
            correct_count, misclassified_count
        )
    )

    logger.info("Okay, great! Now, we want to implement an EDGE HISTOGRAM model.")
    logger.info("We have 784 features for each input, and 784 = 28 * 28. Therefore,")
    logger.info("it is reasonable to assume that these are just greyscale color values")
    logger.info("of tiny handwritten digits. If we can turn them into just 0s and 1s,")
    logger.info("for black and whites respectively, we should be able to eke out some")
    logger.info("better performance. For example, is we had 49 features, we would want")
    logger.info(
        "to turn all the low-greyscale values into 0s so a seven might look like\n"
    )
    logger.info("       ")
    logger.info(" 777777")
    logger.info("    77 ")
    logger.info("   77  ")
    logger.info("  77   ")
    logger.info(" 77    ")
    logger.info("       \n")

    if not os.path.exists(nmist_local_canny_dataset_path):
        logger.warning("NMIST 784 Canny data isn't locally archived! Creating now.")
        # import MNIST
        for i in range(x.shape[1]):
            logger.info("Canny edge-detecting input {}".format(i))
            x[:, i] = (
                feature.canny(x[:, i].reshape(28, 28)).astype(int).reshape(784) * 255
            )
        logger.info("Saving locally to {}".format(nmist_local_canny_dataset_path))
        np.savez(nmist_local_canny_dataset_path, x, y)
        logger.info("Save completed!")
        logger.info("You should now be able to load them back anytime using\n")
        logger.info("    numpy.load('{}')\n".format(nmist_local_canny_dataset_path))
        logger.warning(
            "We will load locally from here on out unless you delete the file."
        )
    else:
        logger.warning(
            "Loading local data from {}...".format(nmist_local_canny_dataset_path)
        )
        npzfile = np.load(nmist_local_canny_dataset_path)
        x = npzfile["arr_0"]
        y = npzfile["arr_1"]
        logger.info("Local data loaded!")

    def our_multiclass_perceptron(w):
        return multiclass_perceptron(w, x, y)

    # We want to initialize a weights matrix of the form (N+1) by C. We know that in
    # this case, C = 10, because there are 10 digits; N+1, meanwhile, is however many
    # features the input data gives us. So here, that should be 785 x 10.
    # weight_matrix = (np.shape(x)[0] + 1, 10)
    weight_matrix = (np.shape(x)[0] + 1, 10)

    logger.info("Generating random starting weights...")
    init_weights = (
        np.random.rand(weight_matrix[0], weight_matrix[1]).astype(np.float32) - 0.5
    )

    logger.info("Minimizing the Multiclass Perceptron cost function via GD...")
    (weights, costs, misses) = gradient_descent(
        our_multiclass_perceptron,
        alpha=0.01,
        max_its=ITERATIONS,
        w=init_weights,
        x=x,
        y=y,
    )
    logger.info("Done!")
    logger.info("Final cost    :: {}".format(costs[-1]))
    logger.info("Final misses  :: {}".format(misses[-1]))

    (correct_count, misclassified_count) = check_classify(weights[-1], x, y)
    logger.warning(
        "MULTICLASS PERCEPTRON RESULTS: {}, {}".format(
            correct_count, misclassified_count
        )
    )

    edge_weights = weights
    edge_costs = costs
    edge_misses = misses

    # Time to create some plots.

    plt.scatter(list(range(len(original_misses))), original_misses)
    plt.scatter(list(range(len(edge_misses))), edge_misses)
    plt.title("Problem 9.2 - Misclassifications, original vs. edge")
    plt.ylabel("Misses")
    plt.xlabel("Iterations")
    plt.draw()
    plt.savefig(os.path.join(figs_dir, "problem_9_2_misses.png"))
    plt.close()
    logger.warning("Misses plot generated.")

    plt.scatter(list(range(len(original_costs))), original_costs)
    plt.scatter(list(range(len(edge_costs))), edge_costs)
    plt.title("Problem 9.2 - Cost functions, original vs. edge")
    plt.ylabel("Costs")
    plt.xlabel("Iterations")
    plt.draw()
    plt.savefig(os.path.join(figs_dir, "problem_9_2_costs.png"))
    plt.close()
    logger.warning("Costs plot generated.")
