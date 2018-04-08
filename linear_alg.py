
"""
Linear algebra routines
"""

import numpy as np


def solve_tridiagonal(A, d):
    """
    :param A: A[0, 0] and A[N-1, N-1] is not used
    """
    assert(A.ndim == 2)
    assert(d.ndim == 1)
    assert(A.shape[0] == len(d))
    assert(A.shape[1] == 3)

    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    N = len(d)

    # forward
    cp = np.zeros(N)
    dp = np.zeros(N)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for idx in range(1, N):
        den = b[idx] - a[idx] * cp[idx - 1]
        cp[idx] = c[idx] / den
        dp[idx] = (d[idx] - a[idx] * dp[idx - 1]) / den

    # backward
    for idx in range(N - 2, -1, -1):
        dp[idx] -= cp[idx] * dp[idx + 1]

    return dp


