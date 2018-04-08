
import numpy as np

from curve import Spline
from diffusion import Diffusion
from mtypes import FDMethod



class Pricer:
    def __init__(self, pde: Diffusion, xs: np.ndarray, ts, lb: callable=None, ub: callable=None):
        self.ts = ts
        self.lb = lb(xs[0], ts)
        self.ub = ub(xs[-1], ts)
        self.xs = xs[1: -1]
        self.pde = pde
        self.state = None
        self.curr = len(ts) - 2

    def set_payout(self, payout: Spline):
        self.state = payout(self.xs)

    def step_back(self, t, method=FDMethod.Explicit):
        while self.curr >= 0 and self.ts[self.curr] > t:
            t0 = self.ts[self.curr]
            t1 = self.ts[self.curr + 1]
            if method == FDMethod.Explicit:
                B = self.pde.get_parameters(t1, t1 - t0, [self.lb[self.curr + 1], self.ub[self.curr + 1]])
