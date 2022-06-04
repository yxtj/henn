# -*- coding: utf-8 -*-
"""
Implement non-linear HE functions via bootstrapping
Function List:
    hemax: max between two numbers
    hemaxl: max among a vector
    relu: ReLU of a number
    sqrt: square root of a number
"""

import numpy as np
from .vectorize import hevectorize

# %% max

def hemax(a, b):
    c = a-b
    z = relu(c)
    return b + z


hemaxl = hevectorize(hemax)
    

# %% univariate functions

def relu(x):
    return x.bootstrap_with_function(lambda x:max(x, 0))
    return max(x, 0)


def sqrt(x):
    return x.bootstrap_with_function(lambda x:np.sqrt(x))
    