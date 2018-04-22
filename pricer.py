
import numpy as np
import abc

from curve import make_linear
from diffusion import Diffusion
from mtypes import FDMethod, BoundType
from linear_alg import solve_axbd
from tradable import Tradable


class Pricer(metaclass=abc.ABCMeta):
    # TODO: should move xs into pde
    @abc.abstractclassmethod
    def price(self, security: Tradable, pde: Diffusion):
        pass


class GridPricer(Pricer):
    conds = {
        BoundType.Dirichlet: {
            "lb": np.array([1, 0, 0]),
            "ub": np.array([0, 0, 1])
        },
        BoundType.Neumann: {
            "lb": np.array([-1, 1, 0]),
            "ub": np.array([0, -1, 1])
        },
        BoundType.Linear: {
            "lb": np.array([1, -2, 1]),
            "ub": np.array([1, -2, 1])
        }
    }

    def __init__(self):
        self.pde = None
        self.xs = None
        self.ts = None
        self.N = 0
        self.bcs = None
        self.lb = None
        self.ub = None
        self.curr = 0
        self.state = None

    def price(self, security: Tradable, pde: Diffusion):
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

        assert security.t_start == 0, "start day of a Tradable should be 0 for pricing"
        self.xs = np.linspace(pde.x_min, pde.x_max, security.steps)  # TODO: why increase grid size get better result?
        self.ts = security.ts
        self.N = len(self.xs)
        assert self.N > 2, "number of steps should be at least 3"
        self.bcs = security.bcs
        self.pde = pde
        self.pde.set_xs(self.xs)  # TODO: xs will be based on both tradable and diffusion, which will be added later
        self.state = np.zeros(self.N - 2)
        assert len(self.ts) > 1, "number of time steps should be at least 2"
        self.curr = len(self.ts) - 1

        # pre-process boundary conditions
        for key, bc in self.bcs.items():
            if bc["type"] == BoundType.Dirichlet or bc["type"] == BoundType.Linear:
                if key == "lb":
                    self.lb = bc["func"](self.xs[0], self.ts)
                elif key == "ub":
                    self.ub = bc["func"](self.xs[-1], self.ts)
            elif bc["type"] == BoundType.Neumann:
                if key == "lb":
                    self.lb = bc["func"](self.xs[0], self.ts) * (self.xs[1] - self.xs[0])
                elif key == "ub":
                    self.ub = bc["func"](self.xs[-1], self.ts) * (self.xs[-1] - self.xs[-2])
            else:
                raise RuntimeError("Unrecognized boundary condition type")

        # pricing
        for t, actions in reversed(security.events):
            self._step_back(t, FDMethod.CN)  # step back to the time of the event
            for action_type, action in actions:
                if action_type == "payout":
                    self.state += action(self.xs[1: -1], self.ts[self.curr])
                elif action_type == "mask":
                    self.state = action(self.state, self.xs[1: -1], self.ts[self. curr])
                else:
                    raise RuntimeError("Unrecognized event type")
        self._step_back(0, FDMethod.CN)
        return self._get_price(pde.x0)

    def get_prices(self):
        return self.state.copy()

    def _step_back(self, t, method=FDMethod.Explicit):
        while self.curr > 0 and self.ts[self.curr] > t:
            t0 = self.ts[self.curr - 1]
            t1 = self.ts[self.curr]
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

    def _get_price(self, x):
        spline = make_linear(self.xs[1: -1], self.state)
        return spline(x)

    def _update_boundary_condition(self, A=None, B=None):
        if B is not None:
            e2 = np.zeros(2)
            for key, bc in self.bcs.items():
                if key == "lb":
                    e2[0] = self._solve_boundary(B[0, :], self.conds[bc["type"]]["lb"], self.lb[self.curr], True)
                elif key == "ub":
                    e2[1] = self._solve_boundary(B[-1, :], self.conds[bc["type"]]["ub"], self.ub[self.curr], False)
                else:
                    raise RuntimeError("Unrecognized bound")
        else:
            e2 = None

        if A is not None:
            e1 = np.zeros(2)
            for key, bc in self.bcs.items():
                if key == "lb":
                    e1[0] = self._solve_boundary(A[0, :], self.conds[bc["type"]]["lb"], self.lb[self.curr - 1], True)
                elif key == "ub":
                    e1[1] = self._solve_boundary(A[-1, :], self.conds[bc["type"]]["ub"], self.ub[self.curr - 1], False)
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
        return ratio * val
