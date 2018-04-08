
"""
Curve classes
"""

import abc
import numpy as np
from typing import Union

import linear_alg as la


class Curve(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def __call__(self, item):
        pass


class Spline(Curve):
    def __init__(self, xs, coefs):
        assert(xs.ndim == 1)
        assert(coefs.ndim == 2)
        assert(xs.shape[0] + 1 == coefs.shape[0])

        self.xs = xs
        self.coefs = coefs
        self.ys = coefs[:, 0]  # the first coefficient should be y itself

    def __call__(self, xs: Union[float, list, np.ndarray]):
        if isinstance(xs, (int, float)):
            return self._get_single(xs)

        idx = np.searchsorted(self.xs, xs)
        dx = xs - self.xs[np.maximum(0, idx - 1)]
        basis = np.ones(len(xs))
        result = np.zeros(len(xs))
        for c in self.coefs[idx].T:
            result += c * basis
            basis *= dx
        return result

    def get_derivative(self, order=1):
        assert(order > 0)
        assert(self.coefs.shape[1] > order)

        new_coefs = []
        for level in range(order, self.coefs.shape[1]):
            m = np.prod(range(level - order + 1, level + 1))
            new_coefs.append(m * self.coefs[:, level])
        new_coefs = np.array(new_coefs).T
        return Spline(self.xs, new_coefs)

    def _get_single(self, x: float):
        idx = np.searchsorted(self.xs, x)
        dx = x - self.xs[max(0, idx - 1)]
        basis = 1
        result = 0
        for c in self.coefs[idx]:
            result += c * basis
            basis *= dx
        return result


def make_linear(xs, ys, left_grad=0, right_grad=0):
    if isinstance(xs, (int, float)):
        xs = np.array([xs])
    if isinstance(ys, (int, float)):
        ys = np.array([ys])

    coefs = np.zeros([len(xs) + 1, 2])
    coefs[1:, 0] = ys
    coefs[0, 0] = coefs[1, 0]
    coefs[1: -1, 1] = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    coefs[0, 1], coefs[-1, 1] = left_grad, right_grad
    return Spline(xs, coefs)


def make_cubic(xs, ys) -> Spline:
    assert(isinstance(xs, np.ndarray))
    assert(isinstance(ys, np.ndarray))
    assert(xs.ndim == 1)
    assert(ys.ndim == 1)
    assert(xs.shape[0] == ys.shape[0])

    N = len(xs)
    coefs = np.zeros([N + 1, 4])  # this includes rightmost spline too, leftmost spline will be added later
    A = np.zeros([N, 3])
    d = np.zeros(N)

    h = xs[1:] - xs[:-1]

    A[0, 1] = 1
    A[-1, 1] = 1
    A[1: -1, 0] = h[:-1]
    A[1: -1, 1] = 2 * (h[:-1] + h[1:])
    A[1: -1, 2] = h[1:]

    ys_diff = ys[1:] - ys[:-1]
    d[1:-1] = 3 * (ys_diff[1:] / h[1:] - ys_diff[:-1] / h[:-1])

    coefs[1:, 0] = ys
    coefs[1:, 2] = la.solve_tridiagonal(A, d)
    coefs[1: -1, 1] = ys_diff / h - h * (coefs[2:, 2] + 2 * coefs[1: -1, 2]) / 3
    coefs[1: -1, 3] = (coefs[2:, 2] - coefs[1: -1, 2]) / 3 / h

    # add left spline
    coefs[0, 0] = coefs[1, 0]
    coefs[0, 1] = coefs[1, 1]

    # add right spline
    dx = h[-1]
    coefs[-1, 1] = coefs[-2, 1] + 2 * coefs[-2, 2] * dx + 3 * coefs[-2, 3] * dx * dx
    return Spline(xs, coefs)
