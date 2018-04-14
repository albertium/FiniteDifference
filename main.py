
import math
from tradable import Bond
from calibrator import calibrate_hull_white
from diffusion import HullWhitePDE
from pricer import GridPricer


zc_bonds = [Bond(1, math.exp(-0.015)), Bond(2, math.exp(-0.02 * 2)), Bond(3, math.exp(-0.025 * 3))]
pde = HullWhitePDE(0.01, -0.3, 0.3)
pricer = GridPricer()
calibrate_hull_white(pde, pricer, zc_bonds)

for zc in zc_bonds:
    res = pricer.price(zc, pde)
    print(res, " ", zc.price)
