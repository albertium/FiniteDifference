
import numpy as np
import curve
import utils
from linear_alg import solve_axbd
from pricer import Pricer
from diffusion import VasicekPDE, BlackScholesPDE
from mtypes import FDMethod, BoundType, OptionType
from pricer_helper import get_vasicek_bond_price, get_black_scholes_price
from tradable import Bond, Option


a = [1, 2, 3]
print(np.searchsorted(a, 0.9))