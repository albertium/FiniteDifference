
# import numpy as np
from scipy import optimize
from typing import List

from pricer import Pricer
from diffusion import HullWhitePDE
from tradable import Tradable


def calibrate_hull_white(pde: HullWhitePDE, pricer: Pricer, zc_bonds: List[Tradable]):
    for zc in zc_bonds:
        assert zc.price is not None

    zc_bonds = sorted(zc_bonds, key=lambda zc: zc.t_end)  # sort by maturity
    maturities = [zc.t_end for zc in zc_bonds]
    pde.reset_theta(maturities)

    def loss(level, idx, zc_bond):
        pde.theta[idx] = level
        res = pricer.price(zc_bond, pde)
        return res - zc_bond.price

    for idx, zc in enumerate(zc_bonds):
        optimize.brentq(loss, -2, 2, args=(idx, zc), xtol=1E-7)  # recall pricer default to 1E-7 precision in unit test
