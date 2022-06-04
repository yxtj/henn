# -*- coding: utf-8 -*-

__doc__ = """
Basic operations for HE numbers
Type 1 (sum over vector): hesum, heavg, hevar, hesum
Type 2 (dot-product over matrix): dot_product, dot_product_11, dot_product_12, dot_product_21, dot_product_22
Type 3.1 (non-linear univariate): relu, (sign, sqrt, tanh)
Type 3.2 (non-linear bivariate): max, (min)
"""

# implement type 1 and 2
from .basic import *

# vectorize a binary HE function.
from .vectorize import hevectorize

# implement type 3 with polynomial approxmiation
#from .nl_polynomial import *

# implement type 3 with bootstrapping
#from .nl_bootstrap import *
