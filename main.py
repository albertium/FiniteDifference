
import numpy as np
import curve
import utils
from linear_alg import solve_axbd
from pricer import Pricer
from diffusion import VasicekPDE
from mtypes import FDMethod, BoundType
from pricer_helper import get_vasicek_bond_price

#
# pde = VasicekPDE(0.05, 0.01 * 1, 1, 0.07)
# xs = np.linspace(-0.3, 0.3, 101)
# ts = np.linspace(0, 1, 41)
# bc = {
#     "lb": {
#         "type": BoundType.Neumann,
#         "func": lambda x, t: np.ones_like(t)
#     },
#     "ub": {
#         "type": BoundType.Neumann,
#         "func": lambda x, t: np.ones_like(t)
#     }
# }
#
# pricer = Pricer(pde, xs, ts, bc)
# pricer.set_payout(lambda x: np.ones_like(x))
# pricer.step_back(0, FDMethod.CN)
# res = pricer.get_price(0.05)
# ans = get_vasicek_bond_price(0.05, 0.01, 1, 0.07, 1)
# print(res)
# print(ans)
# print(res / ans - 1)
# print(pricer.get_prices())


class A:
    def __init__(self):
        self.a = 2
    a = 1


class B(A):
    def __init__(self):
        print(self.a)
        super().__init__()
        print(self.a)


b = B()