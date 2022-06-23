# -*- coding: utf-8 -*-
"""
Implement non-linear HE functions via bootstrapping
Function List:
    hemax: max between two numbers
    hemaxl: max among a vector
    relu: ReLU of a number
    sqrt: square root of a number
    square: square of a number
"""

#import numpy as np
from .vectorize import hevectorize
from math import sqrt as math_sqrt
import time

# %% max

def hemax(a, b):
    c = a-b
    z = relu(c)
    return b + z


hemaxl = hevectorize(hemax)
    

# %% univariate functions

def relu(x):
    #return x.bootstrap_with_function(lambda x:np.maximum(x, 0))
    return x.bootstrap_with_function(lambda x:max(x, 0))


def sqrt(x):
    #return x.bootstrap_with_function(lambda x:np.sqrt(x))
    return x.bootstrap_with_function(lambda x:math_sqrt(x))


def square(x):
    return x.bootstrap_with_function(lambda x:x*x)
    
    
def dummy(x):
    time.sleep(0.01)
    return x
