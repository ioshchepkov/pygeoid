"""Topographic reduction in gravity and geoid modelling"""

import numpy as np

from pygeoid.conventions import units as u
from pygeoid.conventions.constants import G


@u.quantity_input
def bouguer_plate(height: u.m, density: u.kg / u.m**3 = 2670 * u.kg / u.m**3) -> u.mGal:
    r"""Return an attraction of an infinite Bouguer plate.

    Parameters
    ----------
    height : ~pygeoid.conventions.units.Quantity
        Height above sea level.
    density : ~pygeoid.conventions.units.Quantity
        Density of the prism. Default is 2670 kg/m**3.

    Notes
    -----
    .. math::
        F_B = 2\pi G\delta H,

    where :math:`G` -- gravitational constant, :math:`\delta` -- density,
    :math:`H` -- height above sea level.
    """
    return 2 * np.pi * G * density * height


@u.quantity_input
def spherical_bouguer_cap(
    height: u.m, density: u.kg / u.m**3 = 2670 * u.kg / u.m**3
) -> u.mGal:
    r"""Return spherical Bouguer correction.

    Parameters
    ----------
    height : ~pygeoid.conventions.units.Quantity
        Height above sea level.
    density : ~pygeoid.conventions.units.Quantity
        Density of the prism. Default is 2670 kg/m**3.

    Notes
    -----
    The corected (spherical) Bouguer attraction accounts the curvature of the Earth.
    It is calclated by the closed-form formula for a spherical cap of radius
    166.7 km:

    .. math::
        F_{SB} = 2\pi G ((1 + \mu) H - \lambda R),

    where :math:`G` -- gravitational constant, :math:`\delta` -- density,
    :math:`H` -- height above sea level,
    :math:`\lambda` and :math:`\mu` -- dimensionless coefficients,
    :math:`R = R_e + H` -- sum of the mean radius of the Earth and the height.

    References
    ----------
    - LaFehr, T.R., 1991. An exact solution for the gravity
      curvature (Bullard B) correction. Geophysics, 56(8), pp.1179-1184.
    """

    # normal radius
    R0 = 6371 * u.km  # km
    # Bullard B surfase radius
    S = 166.735 * u.km  # km

    alpha = S.value / R0.value
    R = R0 + height

    delta = R0 / R
    eta = height / R
    mu = 1 / 3 * eta**2 - eta

    d = 3 * np.cos(alpha) ** 2 - 2
    f = np.cos(alpha)
    k = np.sin(alpha) ** 2
    p = -6 * np.cos(alpha) ** 2 * np.sin(alpha / 2) + 4 * np.sin(alpha / 2) ** 3
    m = -3 * np.sin(alpha) ** 2 * np.cos(alpha)
    n = 2 * (np.sin(alpha / 2) - np.sin(alpha / 2) ** 2)

    sqrt_f_delta = np.sqrt((f - delta) ** 2 + k)
    llambda_1 = (d + f * delta + delta**2) * sqrt_f_delta + p
    llambda_2 = m * np.log(n / (f - delta + sqrt_f_delta))
    llambda = 1 / 3 * (llambda_1 + llambda_2)

    out = 2 * np.pi * G * density * ((1 + mu) * height - llambda * R)

    return out
