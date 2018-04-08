
"""
helper functions
"""

from scipy.stats import norm
import math


def get_black_scholes_price(S, K, r, q, sig, T):
    d1 = (math.log(S / K) + (r - q) * T) / sig / math.sqrt(T) + 0.5 * sig * math.sqrt(T)
    d2 = d1 - sig * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)