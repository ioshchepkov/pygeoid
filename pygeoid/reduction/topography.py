"""Topographic reduction in gravity and geoid modelling"""

import numpy as np
from pygeoid.constants import _2piG
from pygeoid.units import units as u


def bouguer_plate(height, density=2670, units='mGal'):
    r"""Return an attraction of an infinite Bouguer plate.

    Parameters
    ----------
    height : float or array_like of floats
        Height above sea level, in metres.
    density : float
        Density of the prism, in kg/m**3. Default is 2670 kg/m**3.
    units : str
        Units in which attraction is returned. Default is mGal.

    Notes
    -----
    .. math::
        F_B = 2\pi G\delta H,

    where :math:`G` -- gravitational constant, :math:`\delta` -- density,
    :math:`H` -- height above sea level.
    """
    height = height * u.m
    density = density * (u.kg / u.m**3)
    return (_2piG * density * height).to(units).magnitude


def spherical_bouguer_cap(height, density=2670, units='mGal'):
    r"""Return spherical Bouguer correction.

    Parameters
    ----------
    height : float or array_like of floats
        Height above sea level, in metres.
    density : float
        Density of the prism, in kg/m**3. Default is 2670 kg/m**3.
    units : str
        Units in which attraction is returned. Default is mGal.

    Notes
    -----
    The corected (spherical) Bouguer attraction accounts the curvature of the Earth.
    It is calclated by the closed-form formula for a spherical cap of radius
    166.7 km [1_]:

    .. math::
        F_B = 2\pi G ((1 + \mu) H - \lambda R),

    where :math:`G` -- gravitational constant, :math:`\delta` -- density,
    :math:`H` -- height above sea level,
    :math:`\lambda` and :math:`\mu` -- dimensionless coefficients,
    :math:`R` -- sum of the mean radius of the Earth and the height.

    References
    ----------
    .. [1] LaFehr, T.R., 1991. An exact solution for the gravity
    curvature (Bullard B) correction. Geophysics, 56(8), pp.1179-1184.
    """
    height = height * u.m
    density = density * (u.kg / u.m**3)

    # normal radius
    R0 = 6371 * u.km  # km
    # Bullard B surfase radius
    S = 166.735 * u.km  # km

    alpha = S / R0
    R = R0 + height

    delta = R0 / R
    eta = height / R
    mu = 1 / 3 * eta**2 - eta

    d = 3 * np.cos(alpha)**2 - 2
    f = np.cos(alpha)
    k = np.sin(alpha)**2
    p = -6 * np.cos(alpha)**2 * np.sin(alpha / 2) + 4 * np.sin(alpha / 2)**3
    m = -3 * np.sin(alpha)**2 * np.cos(alpha)
    n = 2 * (np.sin(alpha / 2) - np.sin(alpha / 2)**2)

    sqrt_f_delta = np.sqrt((f - delta)**2 + k)
    llambda_1 = (d + f * delta + delta**2) * sqrt_f_delta + p
    llambda_2 = m * np.log(n / (f - delta + sqrt_f_delta))
    llambda = 1 / 3 * (llambda_1 + llambda_2)

    out = _2piG * density * ((1 + mu) * height - llambda * R)

    return (out).to(units).magnitude
