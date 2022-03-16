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


# %% dot product

def dot_product(a, b):
    if a.ndim == 1:
        if b.ndim == 1:
            return dot_product_11(a,b)
        else:
            return dot_product_12(a, b)
    else:
        if b.ndim == 1:
            return dot_product_21(a,b)
        else:
            return dot_product_22(a, b)

def dot_product_11(w, x):
    assert x.ndim == w.ndim == 1
    assert len(x) == len(w)
    y = x*w
    return sum_list(y)

def dot_product_21(m, x):
    assert m.ndim == 2
    assert x.ndim == 1
    assert m.shape[1] == len(x)
    out = np.array([ dot_product_11(m[i], x) for i in range(m.shape[0]) ])
    return out

def dot_product_12(x, m):
    assert x.ndim == 1
    assert m.ndim == 2
    assert len(x) == m.shape[0]
    out = np.array([ dot_product_11(x, m[:,i]) for i in range(m.shape[1]) ])
    return out

def dot_product_22(a, b):
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            out[i,j] = dot_product_11(a[i], b[:,j])
    return out

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
    