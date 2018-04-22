

import abc
import numpy as np
from typing import List, Union
from copy import deepcopy

from curve import make_linear
from mtypes import OptionType, BoundType, SwapType


class Tradable(metaclass=abc.ABCMeta):
    bc_template = {
        "Standard Neumann": {
            "lb": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t, dtype=float)
            },
            "ub": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t, dtype=float)
            }
        },
        "Standard Linear": {
            "lb": {
                "type": BoundType.Linear,
                "func": lambda x, t: np.zeros_like(t, dtype=float)
            },
            "ub": {
                "type": BoundType.Linear,
                "func": lambda x, t: np.zeros_like(t, dtype=float)
            }
        }
    }

    default_bcs = bc_template["Standard Linear"]

    def __init__(self, events: Union[List[tuple], tuple], t_start, t_end, dt, bcs=None, steps=101, price=None, has_masks=None):
        self.ts = None
        self.events = events if isinstance(events, list) else [events]  # sum of payouts and masks
        self._t_start = t_start
        self._t_end = t_end
        self.dt = dt
        self._sort_events()
        self.has_masks = has_masks if has_masks is not None else self._check_mask()

        self.bcs = bcs if bcs is not None else self.default_bcs
        self._steps = steps
        self.price = price

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_end(self):
        return self._t_end

    @property
    def steps(self):
        return self._steps

    def __add__(self, other):
        assert not self.has_masks and not other.has_masks
        assert self._t_start == other.t_start
        return Tradable(Tradable._merge_events(self.events, other.events),
                        self._t_start, max(self._t_end, other.t_end), min(self.dt, other.dt),
                        Tradable._merge_bcs(self.bcs, other.bcs), max(self.steps, other.steps),
                        Tradable._add_prices(self.price, other.price), has_masks=False)

    def __sub__(self, other):
        assert not self.has_masks and not other.has_masks

        # need to use decorator to protect the state of the action function to be negate
        def negate_action(func):
            def wrapper(x, t):
                return -func(x, t)
            return wrapper

        other_events = []
        # negate payouts
        for time, actions in other.events:
            tmp = []
            for _, action in actions:
                tmp.append(("payout", negate_action(action)))  # has to be payout since no masks
            other_events.append((time, tmp))

        return Tradable(Tradable._merge_events(self.events, other_events),
                        self._t_start, max(self._t_end, other.t_end), min(self.dt, other.dt),
                        Tradable._merge_bcs(self.bcs, other.bcs), max(self.steps, other.steps),
                        Tradable._add_prices(self.price, other.price), has_masks=False)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)

        # need to use decorator to protect the state of the action function to be negate
        def multiply_action(func, m):
            def wrapper(x, t):
                return m * func(x, t)
            return wrapper

        events = []
        for time, actions in self.events:
            tmp = []
            for _, action in actions:
                tmp.append(("payout", multiply_action(action, other)))  # has to be payout since no masks
            events.append((time, tmp))

        return Tradable(events, self._t_start, self._t_end, self.dt, self.bcs, self.steps, self.price, self.has_masks)

    def __matmul__(self, other):
        assert self.has_masks or other.has_masks, "If both are pure payouts, should use +, -, * instead of chaining"
        assert self._t_end == other.t_start
        return Tradable(Tradable._merge_events(other.events, self.events),  # ordering matter here
                        self._t_start, other.t_end, min(self.dt, other.dt),
                        Tradable._merge_bcs(self.bcs, other.bcs), max(self.steps, other.steps),
                        Tradable._add_prices(self.price, other.price), has_masks=self.has_masks or other.has_masks)

    def _sort_events(self):
        # make actions are list
        self.events = [(time, actions if isinstance(actions, list) else [actions]) for time, actions in self.events]
        self.events = sorted(self.events, key=lambda x: x[0])
        assert self.events[-1][0] == self._t_end
        assert self.events[0][0] >= self._t_start

        tmp = [self._t_start] + [x[0] for x in self.events] + [self._t_end]
        self.ts = []
        for start, stop in zip(tmp[:-1], tmp[1:]):
            self.ts.append(np.linspace(start, stop, np.ceil((stop - start) / self.dt)))
        self.ts = np.hstack(self.ts)

    def _check_mask(self):
        for time, actions in self.events:
            for action in actions:
                if action[0] == "mask":
                    return True
        return False

    @staticmethod
    def _merge_events(events1: List[tuple], events2: List[tuple]):
        # use round to allow float numbers comparison
        sorter = {round(time, 10): deepcopy(actions) for time, actions in events1}
        for time, actions in events2:
            key = round(time, 10)
            if key in sorter:
                sorter[key] += actions
            else:
                sorter[key] = actions
        return sorted([(k, v) for k, v in sorter.items()], key=lambda x: x[0])

    @staticmethod
    def _merge_bcs(bcs1, bcs2):
        if bcs1 == bcs2 == Tradable.default_bcs:
            return None  # return default bcs
        if bcs1 != Tradable.default_bcs:
            return bcs1
        return bcs2

    @staticmethod
    def _add_prices(p1, p2):
        if p1 is not None and p2 is not None:
            return p1 + p2
        return None


class HeatSecurity(Tradable):
    def __init__(self, T):
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

        def payout(x, t): return np.exp(x)
        events = (T, ("payout", payout))
        super().__init__(events, 0, T, dt=1/9, bcs=bcs, steps=9)


class Underlying(Tradable):
    def __init__(self, t_end):
        events = (t_end, ("payout", lambda x, t: x))
        # use payout because it dpends only on the underlying x but not state v
        super().__init__(events, t_start=t_end, t_end=t_end, dt=t_end)  # dt is set to the largest possible number


class Option(Tradable):
    def __init__(self, K, t_start, t_end, dt, option_type, KO=None, KI=None, is_american=False, price=None):
        if option_type == OptionType.Call:
            payout = make_linear(K, 0, right_grad=1)
        elif option_type == OptionType.Put:
            payout = make_linear(K, 0, left_grad=-1)
        else:
            raise RuntimeError("Unrecognized option type")

        # add events
        events = (t_end, ("mask", lambda v, x, t: payout(v)))

        if is_american:
            if option_type == OptionType.call:
                mask = lambda v, x, t: np.maximum(v, x - K)
            else:
                mask = lambda v, x, t: np.maximum(v, K - x)
        else:
            mask = None

        bcs = {
            "lb": {
                "type": BoundType.Dirichlet,
                "func": lambda x, t: np.zeros_like(t, dtype=float)
            },
            "ub": {
                "type": BoundType.Neumann,
                "func": lambda x, t: np.ones_like(t, dtype=float)
            }
        }

        if option_type == OptionType.Put:
            bcs["lb"], bcs["ub"] = bcs["ub"], bcs["lb"]

        super().__init__(events, t_start, t_end, dt, bcs, steps=601, price=price)
        self.K = K


class ZCBond(Tradable):
    """
    zero coupon bond
    """
    def __init__(self, t_start, t_end, price=None):
        def payout(x, t): return np.ones_like(x, dtype=float)
        events = [(t_end, [("payout", payout)])]
        super().__init__(events, t_start, t_end, dt=1 / 41, steps=101, price=price)


class Annuity(Tradable):
    def __init__(self, t_start, t_end, coupon_freq, coupon_rate, price=None):
        dt = 1 / coupon_freq

        def coupon(x, t): return coupon_rate * dt * np.ones_like(x, dtype=float)
        events = []
        now = t_end
        while now > t_start:
            events.append((now, [("payout", coupon)]))
            now -= dt
        super().__init__(events, t_start, t_end, 1/41, steps=101, price=price, has_masks=False)


class CBond(Tradable):
    """
    coupon bond
    """
    def __init__(self, t_start, t_end, coupon_freq, coupon_rate, price=None):
        tmp = ZCBond(t_start, t_end) + Annuity(t_start, t_end, coupon_freq, coupon_rate)
        super().__init__(tmp.events, t_start, t_end, 1/41, steps=101, price=price, has_masks=False)


class Swap(Tradable):
    def __init__(self, t_start, t_end, coupon_freq, coupon_rate, swap_type=SwapType.Payer, price=None):
        if swap_type == SwapType.Payer:
            tmp = ZCBond(0, t_start) - ZCBond(0, t_end) - Annuity(t_start, t_end, coupon_freq, coupon_rate)
        elif swap_type == SwapType.Receiver:
            tmp = Annuity(t_start, t_end, coupon_freq, coupon_rate) - ZCBond(0, t_start) + ZCBond(0, t_end)
        else:
            raise RuntimeError("Unrecognized swap type")
        super().__init__(tmp.events, t_start, t_end, 1/41, steps=101, price=price, has_masks=False)