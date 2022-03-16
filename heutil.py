# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 18:43:10 2022

@author: yanxi
"""
from hesign import sign
import numpy as np

#%% sum

def sum_list(x):
    if not isinstance(x, (list, np.ndarray)):
        return input
    n = len(x)
    if n == 1:
        return x[0]
    elif n == 2:
        return x[0] + x[1]
    else:
        return sum_list(x[:n//2]) + sum_list(x[n//2:])
    

def avg_list(x):
    n = len(x)
    f = 1.0/n
    return sum_list(x) * f


def var_list(x, mean=None):
    y = x*x
    m2 = sum_list(y)
    if mean is None:
        mean = avg_list(x)
    return m2 - mean*mean
    
def dot_product(x, w):
    assert x.ndim == w.ndim == 1
    assert len(x) == len(w)
    y = x*w
    return sum_list(y)



# %% max with polynomial approximation

def max(a, b, alpha=7, quick=False):
    x = a+b
    y = a-b
    s = sign(y)
    return ( x + s*y) / 2

def max_list(x, alpha=7, quick=False):
    n = len(x)
    if n == 1:
        return x[0]
    elif n == 2:
        return max(x[0], x[1], alpha, quick)
    else:
        p = n//2
        return max(max_list(x[:p], alpha, quick),
                   max_list(x[p:], alpha, quick),
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
    