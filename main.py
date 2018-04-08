
import numpy as np
import curve
import utils
from linear_alg import solve_axbd
from pricer import Pricer
from diffusion import HeatPDE
from mtypes import FDMethod

xs = np.linspace(-2, 2, 9)
ts = np.linspace(0, 1, 9)
pde = HeatPDE()
pricer = Pricer(pde, xs, ts, lambda x, t: np.exp(-1 - t), lambda x, t: np.exp(3 - t))
pricer.set_payout(lambda x: np.exp(x))
pricer.step_back(0, FDMethod.CN)
result = pricer.get_prices()
answer = np.exp(xs[1: -1] + 1)
diff = np.zeros(7)
diff[:7] = result - answer
rmse = np.sqrt(np.mean(np.power(diff, 2)))
print(rmse)


