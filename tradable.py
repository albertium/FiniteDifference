

import abc
import numpy as np

from curve import make_linear
from mtypes import OptionType, BoundType


class Tradable(metaclass=abc.ABCMeta):
    bc_template = {
        "Standard Neumann": {
            "lb": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            },
            "ub": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            }
        }
    }

    def __init__(self, ts, payout, bcs, price=None, mask=None):
        self.ts = ts
        self.payout = payout
        self.bcs = bcs
        self.price = price
        self.mask = mask


class HeatSecurity(Tradable):
    def __init__(self, ts):
        bcs = {
            "lb": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.exp(-1 - t)
            },
            "ub": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.exp(3 - t)
            }
        }

        def payout(x):
            return np.exp(x)

        super().__init__(ts, payout, bcs)


class Option(Tradable):
    def __init__(self, K, ts, type, KO=None, KI=None, is_american=False, price=None):
        if type == OptionType.Call:
            payout = make_linear(K, 0, right_grad=1)
        elif type == OptionType.Put:
            payout = make_linear(K, 0, left_grad=-1)
        else:
            raise RuntimeError("Unrecognized option type")

        if is_american:
            if type == OptionType.call:
                mask = lambda v, x, t: np.maximum(v, x - K)
            else:
                mask = lambda v, x, t: np.maximum(v, K - x)
        else:
            mask = None

        bcs = {
            "lb": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.zeros_like(t)
            },
            "ub": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t)
            }
        }

        if type == OptionType.Put:
            bcs["lb"], bcs["ub"] = bcs["ub"], bcs["lb"]

        super().__init__(ts, payout, bcs, mask=mask, price=price)
        self.K = K


class Bond(Tradable):
    # TODO: allow cash flows
    def __init__(self, T, price=None):
        ts = np.linspace(0, T, 41)

        def payout(x): return np.ones_like(x)
        super().__init__(ts, payout, self.bc_template["Standard Neumann"], price=price)
