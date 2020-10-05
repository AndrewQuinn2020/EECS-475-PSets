#!/usr/bin/python3

# Andrew Quinn
# Dr. Katsaggelos, EECS 375 / 475
# Problem Set 1
# Problem 2.1

# See problem-2-1.png for the problem statement.

import numpy as np
import math
import itertools

def cartesian(arrays, out=None):
    """
    Taken, and only slightly modified for Python 3 syntax, from

        https://stackoverflow.com/a/1235363


    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def g(w):
    """The g(w) function we want."""
    return np.matmul(np.transpose(w), w)


def points_per_dimension(N=2, ppd=100, verbose=False):
    """Returns the number of linspace points we can afford to give each
    dimension, with a cap of only 100 points.


    So, for N=1, we get to use all 100 points on that single dimension;
    for N=2, we need to evenly space our points on a [-1 x -1]^2 grid,
    so each dimension gets only 10 points; etc.

    I chose to floor rather than ceiling these, because I didn't want
    to end up causing a combinatorial explosion on my hardware. This is
    of course the curse of dimensionality being talked about here."""
    raw_cut = np.power(100, 1/N)
    cut = math.floor(raw_cut)
    cuts_total = np.power(cut, N)
    if verbose:
        print("For N={} dimensions, we can slice each dimension".format(N))
        print("floor(100^(1/{})) = floor({}) = {} ways,".format(N, raw_cut, cut))
        print("  for a total of {}^{} = {} points.".format(cut, N, cuts_total))
        print("    Our ideal was {} points.".format(ppd))
    return cut


def linspace_n_dimensions(N, size=2, ideal_size=100, verbose=False):
    """Given a number of dimensions N, a size of one length of the hypercube,
    & an ideal number of return points ideal_size, return a 1D array of the
    possible values we're going to feed in to the quadratics.

    It is *not* necessarily the case that len(linspace_n_dimensions(args)) ==
    ideal_size. For example, there's no way to uniformly split up [0, 1]^3
    into 12 evenly spaced points. You can split each of the 3 dimensions
    twice, to get 2**3 = 8 points overall, or three times for 3**3 = 27 points
    overall, but to get 12, you would need to pick some arbitrary dimension of
    the hypercube to split 3 times, a la linspace(0, 1, 3), and then split the
    other ones 2 times, like linspace(0, 1, 2). And that's just the case where
    such a split is even *possible*.

    We don't pick an arbitrary edge of the hypercube to split more often than
    the others, plus, we round down: [0, 1]^4 would require 1 dimension to be
    split only once (meaning linspace(0, 1, 1)=0.5) if we asked for anything
    less than 2**4=16 evenly spaced points, and in this case, we would just
    return a single array value - 0.5.
    """

    start = -size/2.0
    end   = size/2.0

    if verbose:
        print("We are splitting the [{}, {}]^N hypercube.".format(start, end))
        print("All rotational symmetries will be preserved; that is,")
        print("we will be splitting each axis along the values of")

    pts = points_per_dimension(N=N, ppd=ideal_size)
    spacings = np.linspace(start, end, pts, endpoint=False)
    if len(spacings) > 1:
        spacings = spacings + abs(spacings[0] - spacings[1]) / 2
    else:
        spacings = [0.0]

    if verbose:
        print("    {}".format(spacings))

    # all_spacings = itertools.repeat(spacings, N)

    return spacings

def linspace_points(N=2, size=2, P=100, verbose=False):
    """Returns an array of evenly spaced points on the N-dimensional
    hypercube [-size/2.0, +size/2.0]^N. There are usually *not* P points
    exactly in the return, unless P is itself the N'th power of some
    integer."""

    spacings = linspace_n_dimensions(N=N, size=size, ideal_size=P,
                                     verbose=verbose)

    return cartesian(list(itertools.repeat(spacings, N)))


if __name__ == "__main__":
    print("Andrew Quinn\nEECS 375 - HW 1 - Problem 2.1\n" + ("-" * 80))

    # You need both [[ and ]] to tell Python it's 2D.
    # test = np.array([[2], [3], [0]])
    # print(test)
    # print(np.transpose(test))
    # print(g(test))
    #
    # print(np.linspace(-1, 1, 100))

    for i in range(1, 100):
        print(linspace_points(i))
        print(len(linspace_points(i)))
