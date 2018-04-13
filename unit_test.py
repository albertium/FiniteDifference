
import unittest
import numpy as np

import linear_alg
import curve
from diffusion import HeatPDE, BlackScholesPDE, VasicekPDE
from pricer import Pricer
from pricer_helper import get_black_scholes_price, get_vasicek_bond_price
from mtypes import OptionType
from tradable import Bond, Option, HeatSecurity


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
        pde = HeatPDE(1)

        sec = HeatSecurity(ts)
        pricer = Pricer()
        pricer.price(sec, pde, xs)

        # Crank Nicolson method
        result = pricer.get_prices()
        answer = np.exp(xs[1: -1] + 1)
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.046205)  # explicit threshold: 0.08898, implicit threshold: 0.046205

    def test_european_call(self):
        # TODO: why accuracy is low? should get down to 10e^-5 level. because of boundary condition?
        """
        using Dirichlet lb and Neumann ub
        """
        S, K, r, q, sig, T = 100, 100, 0.05, 0.02, 0.35, 1
        S_max = 300
        tick = 0.5
        xs = np.linspace(0, S_max, int(S_max / tick + 1))
        ts = np.linspace(0, T, 101)

        opt = Option(K, ts, OptionType.Call)
        pde = BlackScholesPDE(S, r, q, sig)

        # pricing
        pricer = Pricer()
        pricer.price(opt, pde, xs)

        # only check prices between [0.8 K, 1.2 K]
        mask = (xs[1: -1] >= 0.8 * K) & (xs[1: -1] <= 1.2 * K)
        result = pricer.get_prices()[mask]
        answer = [get_black_scholes_price(x, K, r, q, sig, T) for x in xs[1: -1][mask]]
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.00025518)

    def test_european_put(self):
        S, K, r, q, sig, T = 100, 100, 0.05, 0.02, 0.35, 1
        S_max = 300
        tick = 0.5
        xs = np.linspace(0, S_max, int(S_max / tick + 1))
        ts = np.linspace(0, T, 101)

        opt = Option(K, ts, OptionType.Put)
        pde = BlackScholesPDE(S, r, q, sig)

        # pricing
        pricer = Pricer()
        pricer.price(opt, pde, xs)

        # only check prices between [0.8 K, 1.2 K]
        mask = (xs[1: -1] >= 0.8 * K) & (xs[1: -1] <= 1.2 * K)
        result = pricer.get_prices()[mask]
        answer = [get_black_scholes_price(x, K, r, q, sig, T, call=False) for x in xs[1: -1][mask]]
        rmse = np.sqrt(np.mean(np.power(result - answer, 2)))
        self.assertLessEqual(rmse, 0.000290493)


class DiffusionTest(unittest.TestCase):
    def test_vasicek(self):
        pde = VasicekPDE(0.05, 0.01 * 1, 1, 0.07)
        xs = np.linspace(-0.3, 0.3, 101)
        zc_bond = Bond(T=1)
        pricer = Pricer()
        res = pricer.price(zc_bond, pde, xs)
        ans = get_vasicek_bond_price(0.05, 0.01, 1, 0.07, 1)
        self.assertLessEqual(abs(res / ans - 1), 7.63009E-7)


if __name__ == "__main__":
    unittest.main()
