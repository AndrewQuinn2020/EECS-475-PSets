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

if __name__ == "__main__":
    logger.info("EECS 475 - Andrew Quinn - Problem 4.5.(c) and (d) - data gen")
    logger.info("-" * (88 - 11))
