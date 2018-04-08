
import numpy as np

from curve import Spline, make_linear
from diffusion import Diffusion
from mtypes import FDMethod, BoundType
from linear_alg import solve_axbd


class Pricer:
    def __init__(self, pde: Diffusion, xs: np.ndarray, ts, lb: callable=None, ub: callable=None):
        self.xs = xs
        self.ts = ts
        self.N = len(xs)
        self.lb = lb(xs[0], ts)
        self.ub = ub(xs[-1], ts)
        self.pde = pde
        self.pde.set_xs(xs)
        self.state = None
        self.curr = len(ts) - 2

    def set_payout(self, payout: callable):
        self.state = payout(self.xs[1:-1])
        self.curr = len(self.ts) - 2

    def step_back(self, t, method=FDMethod.Explicit):
        while self.curr >= 0 and self.ts[self.curr] >= t:
            t0 = self.ts[self.curr]
            t1 = self.ts[self.curr + 1]
            if method == FDMethod.Explicit:
                A = None
                B = np.zeros([self.N - 2, 3])
                self.pde.update_parameters(B, t1, t1 - t0)
                e1 = None
                e2 = (B[0, 0] * self.lb[self.curr + 1], B[-1, 2] * self.ub[self.curr + 1])
            elif method == FDMethod.Implicit:
                A = np.zeros([self.N - 2, 3])
                self.pde.update_parameters(A, t0, t1 - t0, sign=-1)
                B = None
                e1 = (A[0, 0] * self.lb[self.curr], A[-1, 2] * self.ub[self.curr])
                e2 = None
            elif method == FDMethod.CN:
                A = np.zeros([self.N - 2, 3])
                self.pde.update_parameters(A, t0, t1 - t0, theta=0.5, sign=-1)  # implicit
                B = np.zeros_like(A)
                self.pde.update_parameters(B, t1, t1 - t0, theta=0.5)  # explicit
                e1 = (A[0, 0] * self.lb[self.curr], A[-1, 2] * self.ub[self.curr])
                e2 = (B[0, 0] * self.lb[self.curr + 1], B[-1, 2] * self.ub[self.curr + 1])
            else:
                raise RuntimeError("Unrecognized method")

            self.state = solve_axbd(self.state, A=A, B=B, e1=e1, e2=e2)
            self.curr -= 1

    def get_price(self, x):
        spline = make_linear(self.xs[1: -1], self.state)
        return spline(x)

    def get_prices(self):
        return self.state.copy()

    # def _update_boundary_condition(self, type, A=None, B=None):

