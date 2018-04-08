
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


def solve_axbd(d, A=None, B=None, e1=None, e2=None):
    """
    solve Ax = Bd
    """
    assert(not (A is None and B is None))
    assert(d.ndim == 1)

    if B is not None:
        assert(len(d) == B.shape[0] and B.shape[1] == 3)
        result = np.zeros(len(d))
        result[1:] += B[1:, 0] * d[:-1]
        result += B[:, 1] * d
        result[:-1] += B[:-1, 2] * d[1:]
    else:
        result = d

    if e1 is not None:
        assert(len(e1) == 2)
        result[0] -= e1[0]
        result[-1] -= e1[1]

    if e2 is not None:
        assert (len(e2) == 2)
        result[0] += e2[0]
        result[-1] += e2[1]

    if A is not None:
        assert(len(d) == A.shape[0] and A.shape[1] == 3)
        return solve_tridiagonal(A, result)
    return result
