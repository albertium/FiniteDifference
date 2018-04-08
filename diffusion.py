
"""
diffusion classes
"""

import abc
import numpy as np


class Diffusion(metaclass=abc.ABCMeta):
    def __init__(self, xs):
        self.xs = xs
        N = len(xs)
        self.d2x = np.zeros(N - 2)
        self.dx = np.zeros(N - 1)
        self.d2x[1: -1] = xs[2:] - xs[:-2]  # pre-cache to save computation
        self.dx[1: -1] = xs[1:] - xs[:-1]  # two ends are reserved for bounds if needed

    @abc.abstractclassmethod
    def _drift(self, x, t):
        return 0

    @abc.abstractclassmethod
    def _diffusion(self, x, t):
        return 0

    @abc.abstractclassmethod
    def _reaction(self, x, t):
        return 0

    def get_parameters(self, t, dt, bounds=None):
        """
        :param t: this can either be t0 or t1 depending on forward or backward case
        """
        N = len(self.xs)
        coefs = np.zeros([N, 3])
        mu = self._drift(self.xs, t)
        sig2 = self._diffusion(self.xs, t)
        interest = self._reaction(self.xs, t)

        assert(bounds is not None)  # TODO: later to add logic to deal with no boundary
        self.dx[0] = self.xs[0] - bounds[0]
        self.dx[-1] = bounds[1] - self.xs[-1]
        self.d2x[0] = self.xs[1] - bounds[0]
        self.d2x[-1] = bounds[1] - self.xs[-2]

        coefs[:, 0] = dt * (2 * sig2 / self.dx[:-1] - mu) / self.d2x
        coefs[:, 1] = -dt * (interest + 2 * sig2 / self.dx[:-1] / self.dx[1:])
        coefs[:, 2] = dt * (2 * sig2 / self.dx[1:] + mu) / self.d2x


class BlackScholes(Diffusion):
    def __init__(self, r, q, sig, xs):
        super().__init__(xs)
        self.r = r
        self.q = q
        self.sig2 = sig * sig

    def _drift(self, x, t):
        return self.r * x

    def _diffusion(self, x, t):
        return 0.5 * self.sig2 * x * x

    def _reaction(self, x, t):
        return self.r
