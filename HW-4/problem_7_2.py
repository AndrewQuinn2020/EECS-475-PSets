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

    # In this problem, we want to apply the one-versus-all algorithm.
    # We aren't given code to do this with, so what we need to do instead
    # is first run the two-class classifier over and over again, store
    # the results, and then apply the fusion rule afterwards to make everything
    # work.

    logger.debug(y[0])

    # If
