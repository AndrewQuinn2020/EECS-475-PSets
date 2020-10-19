#!/usr/bin/python3

import logging
import os
import pickle
import sys

import colorlog
import numpy as np
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


def get_cfh(cost_history, title_id=None):
    """Given the usual format for cost_history, generate a nice Matplotlib figure
    for it."""
    fig, ax = plt.subplots()

    if title_id:
        ax.set_title("Cost history - $g(w)$ - {}".format(title_id))
    else:
        ax.set_title("Cost history - $g(w)$")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Cost $g(w)$ for iteration $k$")

    ax.plot(cost_history)

    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    logger.info("EECS 475 - Andrew Quinn - Problem 4.5.(c) and (d) - graph maker")
    logger.info("-" * (88 - 11))

    # Generate graphs for 4.5.(c)
    with open(os.path.join(pickle_dir, "4_5_c_cost_history.pickle"), "rb") as fp:
        logger.debug("Loading cost_history from {}".format(fp.name))
        cost_history = pickle.load(fp)
        logger.debug("cost_history = {}".format(cost_history))

    fig, ax = get_cfh(cost_history, title_id="4.5.(c)")
    plt.show()

    plt.savefig(os.path.join(figs_dir, "4_5_c_cfh.png"))
    plt.close()

    # Generate graphs for 4.5.(c)
    with open(os.path.join(pickle_dir, "4_5_d_cost_history.pickle"), "rb") as fp:
        logger.debug("Loading cost_history from {}".format(fp.name))
        cost_history = pickle.load(fp)
        logger.debug("cost_history = {}".format(cost_history))

    fig, ax = get_cfh(cost_history, title_id="4.5.(d)")
    plt.show()

    plt.savefig(os.path.join(figs_dir, "4_5_d_cfh.png"))
    plt.close()
