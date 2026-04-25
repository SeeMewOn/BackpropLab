from enum import Enum, auto


class RegMode(Enum):
    LEAST_SQUARE_REGRESSION = auto()
    LASSO_REGRESSION = auto()
    RIDGE_REGRESSION = auto()
