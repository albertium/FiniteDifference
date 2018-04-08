
import numpy as np
import curve
import utils
from linear_alg import solve_axbd

A = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
B = A

# print(solve_axbd(np.array([1, 2, 3])))
print(solve_axbd(np.array([1, 2, 3]), B=B))
print(solve_axbd(np.array([8, 26, 28]), A=A))
print(solve_axbd(np.array([1, 2, 3]), A=A, B=B))
