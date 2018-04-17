
import math
from tradable import Bond
from calibrator import calibrate_hull_white
from diffusion import HullWhitePDE
from pricer import GridPricer
from utils import plot_lines


zc_bonds = [Bond(1, math.exp(-0.0)), Bond(2, math.exp(-0.05 * 2)), Bond(3, math.exp(-0.07 * 3))]
pde = HullWhitePDE(0.01, -0.3, 0.3)
pricer = GridPricer()
calibrate_hull_white(pde, pricer, zc_bonds)

plot_lines({"theta": pde.theta}, (0, 5))
