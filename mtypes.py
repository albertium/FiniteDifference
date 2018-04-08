
from enum import Enum


class FDMethod(Enum):
    Explicit = 1
    Implicit = 2
    CN = 3


class BoundType(Enum):
    Dirichlet = 1
    Neumann = 2

