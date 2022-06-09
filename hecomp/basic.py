# -*- coding: utf-8 -*-
"""
Implement type 1 (sum) and type 2 (dot-product) HE operations.
"""
import numpy as np

#%% sum

def hesum(x:np.ndarray):
    n = len(x)
    if n > 2:
        return hesum(x[:n//2]) + hesum(x[n//2:])
    else:
        return sum(x) # covers 1 element and 0 element case
    

def heavg(x:np.ndarray):
    n = len(x)
    f = 1.0/n
    return hesum(x) * f


def hevar(x:np.ndarray, mean=None):
    y = x*x
    m2 = hesum(y)
    if mean is None:
        mean = heavg(x)
    return m2 - mean*mean


def hewsum(x:np.ndarray, w:np.ndarray, b=None):
    """
    Weighted summation among input data <x> and weight <w>.
    They should be two vectors of the same length
    If <b> is not None, add it to the result as a constant bias.
    """
    if b is None:
        return hesum(x*w)
    else:
        t = np.concatenate([x*w, [b]])
        return hesum(t)


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
    return hesum(y)

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
