import numpy as np


def linear_to_db_conversion(x):
    return 10 * np.log10(x)
