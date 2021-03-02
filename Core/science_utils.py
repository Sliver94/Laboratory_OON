import numpy as np
import math
from Core.utils import BERt
from Core.utils import Rs
from Core.utils import Bn
from Core.utils import beta2
from Core.utils import number_of_channels
from Core.utils import df
from Core.utils import gamma
from Core.utils import alpha

from scipy import special


def linear_to_db_conversion(x):
    return 10 * np.log10(x)


def alpha_conversion(x):
    return (x / (20 * math.log10(math.e))) / 1000


def db_to_linear_conversion(x):
    return 10 ** (x / 10)


def fixed_rate_condition(x):
    if x >= 2 * (special.erfcinv(2 * BERt) ** 2) * (Rs / Bn):
        return 100
    else:
        return 0


def flex_rate_condition(x):
    if x < 2 * (special.erfcinv(2 * BERt) ** 2) * (Rs / Bn):
        return 0
    elif x < (14 / 3) * (special.erfcinv((3 / 2) * BERt) ** 2) * (Rs / Bn):
        return 100
    elif x < 10 * (special.erfcinv((8 / 3) * BERt) ** 2) * (Rs / Bn):
        return 200
    else:
        return 400


def shannon(x):
    return (2 * Rs * math.log2(1 + x * (Bn / Rs))) * (10 ** (-9))


def eta_nli_generator():
    alpha_linear = alpha_conversion(alpha)
    return 16 / (27 * math.pi) * math.log(((math.pi ** 2) / 2) * ((beta2 * (Rs ** 2)) / alpha_linear) *
                                          (number_of_channels ** ((2 * Rs) / df)), math.e) * ((gamma ** 2) / (4 * alpha_linear * beta2 * (Rs ** 3)))



