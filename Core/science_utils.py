import numpy as np
import math
from Core.utils import BERt
from Core.utils import Rs
from Core.utils import Bn


def linear_to_db_conversion(x):
    return 10 * np.log10(x)


def fixed_rate_condition(x):
    if x >= 2 * (2 * BERt) * (Rs / Bn):
        return 100
    else:
        return 0


def flex_rate_condition(x):
    if x < 2 * (2 * BERt) * (Rs / Bn):
        return 0
    elif x < (14 / 3) * ((3 / 2) * BERt) * (Rs / Bn):
        return 100
    elif x < 10 * ((8 / 3) * BERt) * (Rs / Bn):
        return 200
    else:
        return 400


def shannon(x):
    return (2 * Rs * math.log2(1 + x * (Bn / Rs))) * (10 ** (-6))
