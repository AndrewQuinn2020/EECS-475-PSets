#!/usr/bin/python3

# canny_quick_and_dirty.py

# Some quick code to illustrate for my classmates how to do edge detection.
import autograd.numpy as np
np.set_printoptions(linewidth=np.inf) # makes it easier to read if you wanna run yourself

# This is from the Jupyter notebook snippet in the professor's repo.
from sklearn.datasets import fetch_openml

print("Fetching NMIST 784... (This might take a while!)")
x, y = fetch_openml("mnist_784", version=1, return_X_y=True)
print("... Fetch completed!\n\n\n")
# re-shape input/output data
x = x.T
y = np.array([int(v) for v in y])[np.newaxis, :]

# THIS is the good s**t
from skimage import feature
print(x[:, 0])
print(x[:, 0].reshape(28, 28))
# What we're doing is first reshaping to a square, then calling the Canny edge detector
# on it to JUST keep the edges, and then finally reshaping it back to the 1D array.
print(feature.canny(x[:, 0].reshape(28, 28)))
# Canny returns True/False, which makes sense for some applications - but let's turn it
# back into integers so we can visually see we've done the right thing.
print(feature.canny(x[:, 0].reshape(28, 28)).astype(int))
# This style of writing code as long chains of methods is called a "fluent interface".
print(feature.canny(x[:, 0].reshape(28, 28)).astype(int).reshape(784))

# This will go through and turn ALL of your input features into 0s and 1s, with 1s where
# the edges are. This will take a while, so uncomment it knowing that.
# for i in range(x.shape[1]):
#     logger.info("Canny edge-detecting input {}".format(i))
#     x[:, i] = feature.canny(x[:, i].reshape(28, 28)).astype(int).reshape(784)
