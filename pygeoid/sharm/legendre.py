"""Legender polynomials and functions"""

import numpy as np
from pyshtools.legendre import PlmBar, PlmBar_d1, PlmIndex


def _lplm_index(lmax):
    """Change index for 1d pyshtools array to 2d array.

    """
    return np.fromfunction(PlmIndex, (lmax + 1, lmax + 1), dtype=int)


def lplm_d1(lmax, x):
    """Associated Legendre functions of the first kind and their derivatives.

    Parameters
    ----------
    lmax : int
        Maximum degree and order.
    x : float
        Input value.

    Returns
    -------
    Plm(x) : (lmax + 1, lmax + 1) array
        Values of Plm(x) for all orders and degrees.
    Plm_d(x) : {(lmax + 1, lmax + 1), None} array
        Values of Plm_d(x) for all orders and degrees.
        Will return None if `x` is very close or equal
        to -1 or 1.
    """
    if not (np.allclose(-1, x) or np.allclose(1, x)):
        index = _lplm_index(lmax)
        p, p_d1 = PlmBar_d1(lmax, x)
        return p[index], p_d1[index]
    else:
        p = lplm(lmax, x)
        return p, None


def lplm(lmax, x):
    """Associated Legendre functions of the first kind.

    Parameters
    ----------
    lmax : int
        Maximum degree and order of the Plm(x).
    x : float
        Input value.

    Returns
    -------
    Plm(x) : (lmax + 1, lmax + 1) array
        Values of Plm(x) for all orders and degrees.
    """
    return PlmBar(lmax, x)[_lplm_index(lmax)]
