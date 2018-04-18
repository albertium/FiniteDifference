
"""
helper functions
"""

import numpy as np
from scipy.stats import norm
import math

from curve import make_cubic
from mtypes import OptionType


def get_black_scholes_price(S, K, r, q, sig, T, call=True):
    d1 = (math.log(S / K) + (r - q) * T) / sig / math.sqrt(T) + 0.5 * sig * math.sqrt(T)
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def get_vasicek_bond_price(r0, theta, kappa, sig, T):
    sig2 = sig * sig
    B = (1 - np.exp(-kappa * T)) / kappa
    A = np.exp((theta / kappa - sig2 / 2 / kappa / kappa) * (B - T) - sig2 / 4 / kappa * B * B)
    return A * np.exp(-B * r0)


def get_vasicek_bond_option_price(K, r0, theta, kappa, sig, T, S, option_type=OptionType.Call):
    """
    From "Bond Option Pricing using the Vasicek Short Rate Model" by Nicholas Burgess
    """
    assert T < S
    P_T = get_vasicek_bond_price(r0, theta, kappa, sig, T)
    P_S = get_vasicek_bond_price(r0, theta, kappa, sig, S)
    B = (1 - np.exp(-kappa * (S - T))) / kappa
    sig_z = sig * np.sqrt((1 - np.exp(-2 * kappa * T)) / 2 / kappa) * B
    h = np.log(P_S / P_T / K) / sig_z + 0.5 * sig_z
    sign = 1 if option_type == OptionType.Call else -1
    return sign * P_S * norm.cdf(sign * h) - sign * K * P_T * norm.cdf(sign * (h - sig_z))


sample_zero_curve = make_cubic(
    xs=np.array([0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35]),
    ys=np.array([0.00544859, 0.00697146, 0.00788897, 0.01306847, 0.02447046, 0.03070799, 0.03350761, 0.03243098,
                 0.02976297, 0.02966949, 0.02928756, 0.02442422])
)