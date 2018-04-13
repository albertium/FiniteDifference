
"""
diffusion classes
"""

import abc
import numpy as np

from curve import Spline, make_cubic


class Diffusion(metaclass=abc.ABCMeta):
    def __init__(self, x0):
        self.x0 = x0
        self.xs = None
        self.dx = None
        self.d2x = None
        self.N = 0

    def set_xs(self, xs):
        self.xs = xs
        self.d2x = xs[2:] - xs[:-2]  # pre-cache to save computation
        self.dx = xs[1:] - xs[:-1]  # two ends are reserved for bounds if needed
        self.N = len(xs)

    @abc.abstractclassmethod
    def _drift(self, x, t):
        return 0

    @abc.abstractclassmethod
    def _diffusion(self, x, t):
        return 0

    @abc.abstractclassmethod
    def _reaction(self, x, t):
        return 0

    def update_parameters(self, coefs, t, dt, theta=1, sign=1):
        """
        :param coefs: pass-in coefs view is updated directly to avoid overhead
        :param t: this can either be t0 or t1 depending on forward or backward case
        :param dt: t1 - t0
        :param theta: as in theta method
        :param sign: default plus
        """
        assert(coefs.ndim == 2)
        assert(coefs.shape == (self.N - 2, 3))
        assert(self.N > 0)  # assert that xs is set

        # only update the interior coefs, boundary are handled outside
        mu = self._drift(self.xs[1: -1], t)  # size of N - 2
        sig2 = self._diffusion(self.xs[1: -1], t)
        interest = self._reaction(self.xs[1: -1], t)

        coefs[:, 0] = sign * theta * dt * (2 * sig2 / self.dx[:-1] - mu) / self.d2x
        coefs[:, 1] = 1 - sign * theta * dt * (interest + 2 * sig2 / self.dx[:-1] / self.dx[1:])
        coefs[:, 2] = sign * theta * dt * (2 * sig2 / self.dx[1:] + mu) / self.d2x


class HeatPDE(Diffusion):
    def _drift(self, x, t):
        return 0

    def _diffusion(self, x, t):
        return 1

    def _reaction(self, x, t):
        return 0


class BlackScholesPDE(Diffusion):
    def __init__(self, S0, r, q, sig):
        super().__init__(S0)
        self.r = r
        self.q = q
        self.sig2 = sig * sig

    def _drift(self, x, t):
        return (self.r - self.q) * x

    def _diffusion(self, x, t):
        return 0.5 * self.sig2 * x * x

    def _reaction(self, x, t):
        return self.r


class VasicekPDE(Diffusion):
    def __init__(self, r0, theta, kappa, sig):
        super().__init__(r0)
        self.r0 = r0
        self.theta = theta
        self.kappa = kappa
        self.sig = sig
        self.sig2 = sig * sig

    def _drift(self, x, t):
        return self.theta - self.kappa * x

    def _diffusion(self, x, t):
        return 0.5 * self.sig2

    def _reaction(self, x, t):
        return x


class HullWhitePDE(Diffusion):
    def __init__(self, zero_curve: Spline, alpha=0.05, sig=0.1):
        super().__init__(0)
        f_tmp = lambda t: zero_curve.get_derivative()(t) * t + zero_curve(t)
        xs = np.linspace(zero_curve.xs[0], zero_curve.xs[-1], 50)
        self.f_curve = make_cubic(xs, f_tmp(xs))
        theta_tmp = lambda t: self.f_curve.get_derivative()(t) + alpha * self.f_curve(t) \
                              + sig * sig * (1 - np.exp(-alpha * t)) / 2 / alpha
        self.theta = make_cubic(xs, theta_tmp(xs))

        self.alpha = alpha
        self.sig = sig
        self.sig2 = sig * sig

    def _drift(self, x, t):
        return self.theta(t) - self.alpha * x

    def _diffusion(self, x, t):
        return 0.5 * self.sig2

    def _reaction(self, x, t):
        return x
