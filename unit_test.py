
import unittest
import numpy as np

import linear_alg
import curve
from diffusion import HeatPDE, BlackScholesPDE, VasicekPDE
from pricer import Pricer
from pricer_helper import get_black_scholes_price, get_vasicek_bond_price
from mtypes import FDMethod, BoundType


class LinearAlgTest(unittest.TestCase):
    def test_solve_tridiagonal(self):
        A = np.array([[0, 1, 1], [1, 2, 3], [2, 1, 0]])
        d = np.array([3, 14, 7])
        result = linear_alg.solve_tridiagonal(A, d)
        self.assertTrue(np.allclose(result, [1, 2, 3]))

    def test_solve_axbd(self):
        mat = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

        # test x = Bd
        res = linear_alg.solve_axbd(np.array([1, 2, 3]), B=mat)
        self.assertTrue(np.allclose(res, [8, 26, 28]))

        # test Ax = d
        res = linear_alg.solve_axbd(np.array([8, 26, 28]), A=mat)
        self.assertTrue(np.allclose(res, [1, 2, 3]))

        # test Ax = Bd
        res = linear_alg.solve_axbd(np.array([1, 2, 3]), A=mat, B=mat)
        self.assertTrue(np.allclose(res, [1, 2, 3]))


class CurveTest(unittest.TestCase):
    def test_make_cubic(self):
        def func(x):
            return np.power(x + 1, 3) - 1

        xs = np.linspace(-10, 10, 20)
        ys = func(xs)
        spline = curve.make_cubic(xs, ys)

        # exact
        result = spline(xs)
        self.assertTrue(np.allclose(result, ys))

        # approximation
        xs = np.linspace(-10, 10, 200)
        answer = func(xs)
        result = spline(xs)
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse / np.mean(np.abs(answer)), 0.005)

    def test_get_derivative(self):
        def func(x):
            return np.power(x + 1, 3) - 1

        def deriv1(x):
            return 3 * np.power(x + 1, 2)

        def deriv2(x):
            return 6 * (x + 1)

        xs = np.linspace(-10, 10, 20)
        ys = func(xs)
        spline = curve.make_cubic(xs, ys)
        d1 = spline.get_derivative(1)
        d2 = spline.get_derivative(2)
        d2b = d1.get_derivative(1)

        # d1 == d1
        xs = np.linspace(-8, 8, 200)
        result = d1(xs)
        answer = deriv1(xs)
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse / np.mean(np.abs(answer)), 0.005)

        # d2 == d2
        xs = np.linspace(-6, 6, 200)
        result = d2(xs)
        answer = deriv2(xs)
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse / np.mean(np.abs(answer)), 0.005)

        # d2 = d2b
        xs = np.linspace(-15, 15, 200)
        result = d2(xs)
        answer = d2b(xs)
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertTrue(np.allclose(result, answer))

        # f'' = 0
        self.assertEqual(d1(-10), d1(-100))


class PricerTest(unittest.TestCase):
    def test_heat_dirichlet(self):
        xs = np.linspace(-2, 2, 9)
        ts = np.linspace(0, 1, 9)
        pde = HeatPDE()
        bc = {
            "type": BoundType.Dirichlet,
            "lb": lambda x, t: np.exp(-1 - t),
            "ub": lambda x, t: np.exp(3 - t)
        }
        pricer = Pricer(pde, xs, ts, bc)

        # explicit method
        pricer.set_payout(lambda x: np.exp(x))
        pricer.step_back(0)
        result = pricer.get_prices()
        answer = np.exp(xs[1: -1] + 1)
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.08898)

        # implicit method
        pricer.set_payout(lambda x: np.exp(x))
        pricer.step_back(0, FDMethod.Implicit)
        result = pricer.get_prices()
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.17202)

        # Crank Nicolson method
        pricer.set_payout(lambda x: np.exp(x))
        pricer.step_back(0, FDMethod.CN)
        result = pricer.get_prices()
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.046205)

    def test_european_call(self):
        # TODO: why accuracy is low? should get down to 10e^-5 level. because of boundary condition?
        S, K, r, q, sig, T = 100, 100, 0.05, 0.02, 0.35, 1
        S_max = 300
        tick = 0.5
        xs = np.linspace(0, S_max, int(S_max / tick + 1))
        ts = np.linspace(0, T, 101)
        pde = BlackScholesPDE(r, q, sig)

        # ========== Call ==========
        # bc
        bc = {
            "lb": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.zeros_like(t)
            },
            "ub": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: x * np.exp(-q * (T - t)) - K * np.exp(-r * (T - t))
            }
        }

        # pricing
        pricer = Pricer(pde, xs, ts, bc)
        payout = curve.make_linear(K, 0, right_grad=1)
        pricer.set_payout(payout)
        pricer.step_back(0, FDMethod.CN)

        # only check prices between [0.8 K, 1.2 K]
        mask = (xs[1: -1] >= 0.8 * K) & (xs[1: -1] <= 1.2 * K)
        result = pricer.get_prices()[mask]
        answer = [get_black_scholes_price(x, K, r, q, sig, T) for x in xs[1: -1][mask]]
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.00029041)

        # ========== Call Neumann ==========
        # bc
        bc = {
            "lb": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.zeros_like(t)
            },
            "ub": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            }
        }

        # pricing
        pricer = Pricer(pde, xs, ts, bc)
        payout = curve.make_linear(K, 0, right_grad=1)
        pricer.set_payout(payout)
        pricer.step_back(0, FDMethod.CN)

        # only check prices between [0.8 K, 1.2 K]
        mask = (xs[1: -1] >= 0.8 * K) & (xs[1: -1] <= 1.2 * K)
        result = pricer.get_prices()[mask]
        answer = [get_black_scholes_price(x, K, r, q, sig, T) for x in xs[1: -1][mask]]
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.00025518)

        # ========== Put ==========
        # bc
        bc = {
            "lb": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: x * np.exp(-q * (T - t))
            },
            "ub": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.zeros_like(t)
            }
        }

        # pricing
        pricer = Pricer(pde, xs, ts, bc)
        payout = curve.make_linear(K, 0, left_grad=-1)
        pricer.set_payout(payout)
        pricer.step_back(0, FDMethod.CN)

        # only check prices between [0.8 K, 1.2 K]
        mask = (xs[1: -1] >= 0.8 * K) & (xs[1: -1] <= 1.2 * K)
        result = pricer.get_prices()[mask]
        answer = [get_black_scholes_price(x, K, r, q, sig, T, call=False) for x in xs[1: -1][mask]]
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.000290493)

        # ========== Put Neumann ==========
        # bc
        bc = {
            "lb": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            },
            "ub": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.zeros_like(t)
            }
        }

        # pricing
        pricer = Pricer(pde, xs, ts, bc)
        payout = curve.make_linear(K, 0, left_grad=-1)
        pricer.set_payout(payout)
        pricer.step_back(0, FDMethod.CN)

        # only check prices between [0.8 K, 1.2 K]
        mask = (xs[1: -1] >= 0.8 * K) & (xs[1: -1] <= 1.2 * K)
        result = pricer.get_prices()[mask]
        answer = [get_black_scholes_price(x, K, r, q, sig, T, call=False) for x in xs[1: -1][mask]]
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.000290493)  # TODO: why this is the same as above?


class DiffusionTest(unittest.TestCase):
    def test_vasicek(self):
        # TODO: check to see if the lower bound Neumann condition implementation is correct
        pde = VasicekPDE(0.05, 0.01 * 1, 1, 0.07)
        xs = np.linspace(-0.3, 0.3, 101)
        ts = np.linspace(0, 1, 41)
        bc = {
            "lb": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            },
            "ub": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            }
        }

        pricer = Pricer(pde, xs, ts, bc)
        pricer.set_payout(lambda x: np.ones_like(x))
        pricer.step_back(0, FDMethod.CN)
        res = pricer.get_price(0.05)
        ans = get_vasicek_bond_price(0.05, 0.01, 1, 0.07, 1)
        self.assertLessEqual(abs(res / ans - 1), 7.63002E-7)


if __name__ == "__main__":
    unittest.main()
