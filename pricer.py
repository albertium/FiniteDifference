
import numpy as np

from curve import make_linear
from diffusion import Diffusion
from mtypes import FDMethod, BoundType
from linear_alg import solve_axbd


class Pricer:
    conds = {
        BoundType.Dirichlet: {
            "lb": np.array([1, 0, 0]),
            "ub": np.array([0, 0, 1])
        },
        BoundType.Neumann: {
            "lb": np.array([-1, 1, 0]),
            "ub": np.array([0, -1, 1])
        }
    }

    def __init__(self, pde: Diffusion, xs: np.ndarray, ts, boundary_condition: dict=None):
        """
        boundary_condition example:
        {
            "lb": {
                "type": "Dirichlet",
                "func": func,
                "idx": 0
            },
            "ub": {
                "type": "Neumann",
                "func": func,
                "idx": -1
            }
        }
        """
        self.xs = xs
        self.ts = ts
        self.N = len(xs)
        self.bc = boundary_condition
        self.pde = pde
        self.pde.set_xs(xs)
        self.state = None
        self.curr = len(ts) - 2

        # pre-process boundary conditions
        for key, bc in boundary_condition.items():
            if bc["type"] == BoundType.Dirichlet:
                if key == "lb":
                    self.lb = bc["func"](xs[0], ts)
                elif key == "ub":
                    self.ub = bc["func"](xs[-1], ts)
            elif bc["type"] == BoundType.Neumann:
                if key == "lb":
                    self.lb = bc["func"](xs[0], ts) * (xs[1] - xs[0])
                elif key == "ub":
                    self.ub = bc["func"](xs[-1], ts) * (xs[-1] - xs[-2])
            else:
                raise RuntimeError("Unrecognized boundary condition type")

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
            elif method == FDMethod.Implicit:
                A = np.zeros([self.N - 2, 3])
                self.pde.update_parameters(A, t0, t1 - t0, sign=-1)
                B = None
            elif method == FDMethod.CN:
                A = np.zeros([self.N - 2, 3])
                self.pde.update_parameters(A, t0, t1 - t0, theta=0.5, sign=-1)  # implicit
                B = np.zeros_like(A)
                self.pde.update_parameters(B, t1, t1 - t0, theta=0.5)  # explicit
            else:
                raise RuntimeError("Unrecognized method")

            e1, e2 = self._update_boundary_condition(A, B)
            self.state = solve_axbd(self.state, A=A, B=B, e1=e1, e2=e2)
            self.curr -= 1

    def get_price(self, x):
        spline = make_linear(self.xs[1: -1], self.state)
        return spline(x)

    def get_prices(self):
        return self.state.copy()

    def _update_boundary_condition(self, A=None, B=None):
        if B is not None:
            e2 = np.zeros(2)
            for key, bc in self.bc.items():
                if key == "lb":
                    e2[0] = self._solve_boundary(B[0, :], self.conds[bc["type"]]["lb"], self.lb[self.curr + 1], True)
                elif key == "ub":
                    e2[1] = self._solve_boundary(B[-1, :], self.conds[bc["type"]]["ub"], self.ub[self.curr + 1], False)
                else:
                    raise RuntimeError("Unrecognized bound")
        else:
            e2 = None

        if A is not None:
            e1 = np.zeros(2)
            for key, bc in self.bc.items():
                if key == "lb":
                    e1[0] = self._solve_boundary(A[0, :], self.conds[bc["type"]]["lb"], self.lb[self.curr], True)
                elif key == "ub":
                    e1[0] = self._solve_boundary(A[-1, :], self.conds[bc["type"]]["ub"], self.ub[self.curr], False)
                else:
                    raise RuntimeError("Unrecognized bound")
        else:
            e1 = None

        return e1, e2

    @staticmethod
    def _solve_boundary(mat, cond, val, left: bool=True):
        """
        helper function for _update_boundary_condition
        :param mat: A[0, -1: 1] or A[-1, n-1: n+1]
        :param cond: for Dirichlet, it would be [1, 0, 0] or [0, 0, 1]
        :param val: bound value
        :param left: eliminate left side or right side
        :return:
        """
        if left:
            ratio = mat[0] / cond[0]
        else:
            ratio = mat[-1] / cond[-1]
        mat -= ratio * cond
        return -ratio * val
