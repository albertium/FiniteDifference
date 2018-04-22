
from enum import Enum


class FDMethod(Enum):
    Explicit = 1
    Implicit = 2
    CN = 3


class BoundType(Enum):
    Dirichlet = 1
    Neumann = 2
    Linear = 3


class OptionType(Enum):
    Call = 1
    Put = 2


class SwapType(Enum):
    Payer = 1
    Receiver = 2
