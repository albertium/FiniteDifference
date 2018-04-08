
import unittest
import numpy as np

import linear_alg
import curve
import utils


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


if __name__ == "__main__":
    unittest.main()
