
import numpy as np
import curve
import utils
from linear_alg import solve_axbd
from pricer import Pricer
from diffusion import HeatPDE, BlackScholesPDE
from mtypes import FDMethod, BoundType
from pricer_helper import get_black_scholes_price


S, K, r, q, sig, T = 100, 100, 0.05, 0.02, 0.35, 1
S_max = 300
tick = 0.5
xs = np.linspace(0, S_max, int(S_max / tick + 1))
ts = np.linspace(0, T, 101)
pde = BlackScholesPDE(r, q, sig)

# bc
bc = {
    "lb": {
        "type": BoundType.Dirichlet,
        "func": lambda x, t: np.ones_like(t) * 100000
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
print(rmse)
print(pricer.get_price(100))






