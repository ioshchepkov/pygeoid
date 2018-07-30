"""
Auxiliarry functions for the pointwise gravity spherical harmonics expansion.
"""

import warnings
import numpy as np
from joblib import Parallel, delayed
from .legendre import lplm, lplm_d1

np.seterr(over='raise')


def expand_parallel(x, q, *args):
    nlat = x.shape[0]
    # parallel only if there are more than one circle
    if nlat > 1:
        values = np.array(Parallel(n_jobs=-1)(delayed(expand_circle)(x[i], q[i], *args)
                                              for i in range(nlat)))
    else:
        values = np.array([expand_circle(x[i], q[i], *args)
                           for i in range(nlat)])
    return values


def expand_circle(x, q, in_coeff_func, sum_func, *args):
    if q.size == 1:
        in_coeff = in_coeff_func(x, q, *args[:-2])

    values = []
    for j in range(args[-2][0].shape[0]):
        if q.size > 1:
            in_coeff = in_coeff_func(x[j], q[j], *args[:-2])

        out = sum_func(in_coeff, args[-2][:, j], args[-1])
        values.append(out)

    return values


def common_precompute(lat, lon, r, r0, lmax):

    lat = np.atleast_2d(lat)
    lon = np.atleast_2d(lon)
    r = np.atleast_2d(r)

    degrees = np.arange(lmax + 1)
    m = np.atleast_2d(lon[0]).T * degrees

    cosin = np.array([np.cos(m), np.sin(m)])

    if np.allclose(r[:, 0, None], r):
        ri = 1 / r[:, 0]
        x = np.sin(lat[:, 0])
    else:
        ri = 1 / r
        x = np.sin(lat)

    q = np.asarray(r0 * ri)

    if np.any(q > 1.01):
        warnings.filterwarnings('once')
        warnings.warn("Possible singularity in downward continuation, r << r0")

    return lat, lon, degrees, cosin, x, q


def in_coeff_potential(x, q, lmax, degrees):
    p = lplm(lmax, x)

    q = np.power(q, degrees)

    l_coeff = np.tile(q, (lmax + 1, 1)).T
    in_coeff = l_coeff * p

    return in_coeff


def sum_potential(in_coeff, cosin, cilm):
    cosm_sinm_sum = cilm[0] * cosin[0] + cilm[1] * cosin[1]
    pot = np.sum(in_coeff * (cosm_sinm_sum))
    return pot


def in_coeff_r_derivative(x, q, lmax, degrees):
    p = lplm(lmax, x)
    q = np.power(q, degrees)

    l_coeff = np.tile(q * (degrees + 1), (lmax + 1, 1)).T
    in_coeff = l_coeff * p

    return in_coeff


def in_coeff_lat_derivative(x, q, lmax, degrees):
    pole = np.allclose(x, -1) | np.allclose(x, 1)
    if not pole:
        _, p_d1 = lplm_d1(lmax, x)
        q = np.power(q, degrees)
        l_coeff = np.tile(q, (lmax + 1, 1)).T
        in_coeff = l_coeff * p_d1
    else:
        in_coeff = 0.0

    return in_coeff


def sum_lat_derivative(in_coeff, cosin, cilm):
    if not np.all(in_coeff == 0.0):
        cosm_sinm_sum = cilm[0] * cosin[0] + cilm[1] * cosin[1]
        lat_d = np.sum(in_coeff * (cosm_sinm_sum))
    else:
        lat_d = 0.0

    return lat_d


def in_coeff_lon_derivative(x, q, lmax, degrees, m_coeff):
    pole = np.allclose(x, -1) | np.allclose(x, 1)
    if not pole:
        p = lplm(lmax, x)
        q = np.power(q, degrees)
        l_coeff = np.tile(q, (lmax + 1, 1)).T
        in_coeff = l_coeff * m_coeff * p
    else:
        in_coeff = 0.0

    return in_coeff


def sum_lon_derivative(in_coeff, cosin, cilm):
    if not np.all(in_coeff == 0.0):
        lon_d = np.sum(in_coeff * (-cilm[1] * cosin[0] +
                                   cilm[0] * cosin[1]))
    else:
        lon_d = 0.0

    return lon_d


def in_coeff_gradient(x, q, lmax, degrees, m_coeff):

    p, p_d1 = lplm_d1(lmax, x)
    q = np.power(q, degrees)

    l_coeff_1 = np.tile(q, (lmax + 1, 1)).T
    l_coeff_rad_d = np.tile(q * (degrees + 1), (lmax + 1, 1)).T

    in_coeff_rad_d = l_coeff_rad_d * p

    if p_d1 is not None:
        in_coeff_lat_d = l_coeff_1 * p_d1
        in_coeff_lon_d = l_coeff_1 * m_coeff * p
    else:
        in_coeff_lat_d = in_coeff_lon_d = 0.0

    in_coeff = (in_coeff_rad_d, in_coeff_lat_d, in_coeff_lon_d)

    return in_coeff


def sum_gradient(in_coeff, cosin, cilm):
    cosm_sinm_sum = cilm[0] * cosin[0] + cilm[1] * cosin[1]

    rad_d = np.sum(in_coeff[0] * (cosm_sinm_sum))

    if not np.all(in_coeff[1:] == 0.0):
        lat_d = np.sum(in_coeff[1] * (cosm_sinm_sum))
        lon_d = np.sum(in_coeff[2] * (-cilm[1] * cosin[0] +
                                      cilm[0] * cosin[1]))
    else:
        lat_d = lon_d = 0.0

    return (lat_d, lon_d, rad_d)


def in_coeff_gravity_anomaly(x, q, lmax, degrees):
    p = lplm(lmax, x)
    q = np.power(q, degrees)

    l_coeff = np.tile(q * (degrees - 1), (lmax + 1, 1)).T
    in_coeff = l_coeff * p

    return in_coeff
