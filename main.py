
import numpy as np
import curve
import utils


def func(x):
    return np.power(x + 1, 3) - 1


def func1(x):
    return 3 * np.power(x + 1, 2)


def func2(x):
    return 6 * (x + 1)


xs = np.linspace(-10, 10, 20)
ys = func(xs)
xs = np.array([5])
ys = np.array([5])
spline = curve.make_linear(5, 5, -1, right_grad=1)
d1 = spline.get_derivative()
# utils.plot_lines({"points": [xs, ys], "spline": spline}, [-15, 15])
utils.plot_lines({"spline": d1}, [-15, 15])