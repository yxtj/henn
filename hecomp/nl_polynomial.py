# -*- coding: utf-8 -*-
"""
Implement non-linear HE functions via polynomial approximation
Function List:
    hemax: max between two numbers
    hemaxl: max among a vector
    relu: ReLU of a number
    sqrt: square root of a number
"""
from hesign import sign
#import numpy as np

# %% max with polynomial approximation

def hemax(a, b, alpha=7, quick=False):
    x = a+b
    y = a-b
    s = sign(y)
    return ( x + s*y) / 2

def hemaxl(x, alpha=7, quick=False):
    n = len(x)
    if n == 1:
        return x[0]
    elif n == 2:
        return hemax(x[0], x[1], alpha, quick)
    else:
        p = n//2
        return hemax(hemaxl(x[:p], alpha, quick),
                     hemaxl(x[p:], alpha, quick),
                     alpha, quick)

# %% univariate functions

def relu(x, alpha=7, quick=False):
    '''
    ReLU approximation using HE-sign function (polynomial approximation)
    '''
    return ( x + sign(x)*x) / 2


def sqrt(x):
    '''
    Square root approximation with polynonimal approximation
    1 + (x-1)/2 - (x-1)**2/8 + (x-1)**3/16 - (x-1)**4*(5/128) + ...
    '''
    t = x-1
    return 1 + t/2 - t*t/8
    

def square(x):
    return x*x
    
