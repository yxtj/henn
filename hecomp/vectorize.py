# -*- coding: utf-8 -*-
"""
Vectorize a binary function for HE (via binary-tree structure)
"""

def hevectorize(bfun):
    """
    Vectorize a binary function for HE (via binary-tree structure).
    Return a vectorized function.
    """
    def vfun(x):
        n = len(x)
        if n > 2:
            p = n//2
            return bfun(vfun(x[:p]), vfun(x[p:]))
        if n == 2:
            return bfun(x[0], x[1])
        elif n == 1:
            return x[0]
    return vfun
