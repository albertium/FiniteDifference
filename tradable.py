

import abc
import numpy as np

from curve import make_linear
from mtypes import OptionType, BoundType


class Tradable(metaclass=abc.ABCMeta):
    bcs = {
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

    def __init__(self, ts, payout, bc, price=None, mask=None):
        self.ts = ts
        self.payout = payout
        self.bc = bc
        self.price = price
        self.mask = mask


class Option(Tradable):
    def __init__(self, K, T, type, KO=None, KI=None, is_american=False, price=None):
        ts = np.linspace(0, T, 41)
        payout = make_linear(K, 0, right_grad=1) if type == OptionType.call else make_linear(K, 0, left_grad=-1)
        if is_american:
            if type == OptionType.call:
                mask = lambda v, x, t: np.maximum(v, x - K)
            else:
                mask = lambda v, x, t: np.maximum(v, K - x)
        else:
            mask = None

        bc = self.bcs["Standard Neumann"]
        super().__init__(ts, payout, bc, mask=mask, price=price)
        self.K = K


class Bond(Tradable):
    # TODO: allow cash flows
    def __init__(self, T, price=None):
        ts = np.linspace(0, T, 41)
        payout = lambda x: np.ones_like(x)
        super().__init__(ts, payout, self.bcs["Standard Neumann"], price=price)