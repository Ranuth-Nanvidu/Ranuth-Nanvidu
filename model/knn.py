"""Pure-Python k-nearest-neighbours. No I/O, so it can be tested on its own.

A point is an (x, y, label) tuple, where label is "blue" or "orange".
"""

import heapq

from .config import K


def nearest(points, x, y, k):
    """The k points closest to (x, y), nearest first. Equal distances keep input order."""
    return heapq.nsmallest(k, points, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)


def predict(points, x, y, k=K):
    """Return (label, p_orange) for (x, y).

    p_orange is the fraction of the k nearest neighbours that are orange.
    A tie at exactly 0.5 goes to the label of the single nearest neighbour.
    """
    if not points:
        raise ValueError("cannot predict with no training points")
    near = nearest(points, x, y, min(k, len(points)))
    p_orange = sum(p[2] == "orange" for p in near) / len(near)
    if p_orange > 0.5:
        return "orange", p_orange
    if p_orange < 0.5:
        return "blue", p_orange
    return near[0][2], p_orange


def confidence(label, p_orange):
    """How sure the model is of `label`, from 0.5 to 1."""
    return p_orange if label == "orange" else 1 - p_orange


def loo_accuracy(points, k=K):
    """Leave-one-out accuracy: predict each point from all the others. None if under 2 points."""
    if len(points) < 2:
        return None
    correct = 0
    for i, (x, y, label) in enumerate(points):
        rest = points[:i] + points[i + 1:]
        correct += predict(rest, x, y, k)[0] == label
    return correct / len(points)
