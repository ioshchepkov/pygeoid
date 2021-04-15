"""Some numerical constants from IERS Conventions 2010.

"""

import numpy as _np
import astropy.units as _u
from astropy.constants import Constant as _Constant

####################################################
# IERS2010 Conventions
####################################################

####################################################
# Natural measurable constants
####################################################

G = _Constant(
    abbrev='G',
    name='Constant of gravitation',
    value=6.67428e-11,
    unit='m**3 / (kg * s**2)',
    uncertainty=6.7e-15,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

####################################################
# Auxiliary defining constants
####################################################

L_G = _Constant(
    abbrev='L_G',
    name='1 - d(TT)/d(TCG)',
    value=6.969290134e-10,
    unit='',
    uncertainty=0,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

####################################################
# Body constants
####################################################

GM_sun = _Constant(
    abbrev='GM_sun',
    name='Heliocentric gravitational constant',
    value=1.32712442099e20,
    unit='m**3 / s**2',
    uncertainty=1e10,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

####################################################
# Earth constants
####################################################

GM_earth_tcg = _Constant(
    abbrev='GM_earth',
    name='Geocentric gravitational constant (TCG-compatible)',
    value=3.986004418e14,
    unit='m**3 / s**2',
    uncertainty=8e15,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

GM_earth_tt = _Constant(
    abbrev='GM_earth',
    name='Geocentric gravitational constant (TT-compatible)',
    value=3.986004415e14,
    unit='m**3 / s**2',
    uncertainty=8e15,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

a = _Constant(
    abbrev='a',
    name='Equatorial radius of the Earth',
    value=6378136.6,
    unit='m',
    uncertainty=0.1,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

J2_earth = _Constant(
    abbrev='J2_earth',
    name='Dynamical form factor of the Earth',
    value=0.0010826359,
    unit='',
    uncertainty=1e-10,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

rf = _Constant(
    abbrev='rf',
    name='Flattening factor of the Earth',
    value=298.25642,
    unit='',
    uncertainty=0.00001,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

ge = _Constant(
    abbrev='ge',
    name='Mean equatorial gravity',
    value=9.7803278,
    unit='m / s**2',
    uncertainty=0.00001,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

W0 = _Constant(
    abbrev='W0',
    name='Potential of the geoid',
    value=62636856.0,
    unit='m**2 / s**2',
    uncertainty=0.5,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

R0 = _Constant(
    abbrev='R0',
    name='Geopotential scale factor (GM/W0)',
    value=6363672.6,
    unit='m',
    uncertainty=0.1,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

H = _Constant(
    abbrev='H',
    name='Dynamical flattening',
    value=3273795e-9,
    unit='',
    uncertainty=1e-9,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')


def tcg_to_tt(x):
    """Convert TCG-compatible value to TT-compatible value.

    """
    return x * (1 - L_G)

####################################################
# Love and Shida numbers of the second-degree
####################################################


k2 = _Constant(
    abbrev='k2',
    name='Nominal degree 2 Love number k2',
    value=0.29525,
    unit='',
    uncertainty=0,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

h2 = _Constant(
    abbrev='h2',
    name='Nominal degree 2 Love number h2',
    value=0.6078,
    unit='',
    uncertainty=0,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

l2 = _Constant(
    abbrev='l2',
    name='Nominal degree 2 Shida number l2',
    value=0.0847,
    unit='',
    uncertainty=0,
    reference='IERS Conventions(2010), '
    'IERS Technical Note 36, '
    'Verlagdes Bundesamts für Kartographie und Geodäsie, '
    'Frankfurt am Main, Germany.')

DEGREE2_LOVE_NUMBERS = {'k': k2, 'l': l2, 'h': h2}


def l2_shida_number(lat: _u.deg = None) -> _u.dimensionless_unscaled:
    """Return degree 2 Shida number (l2,0).

    If `lat` is None, the nominal degree 2 Shida number l2=0.0847
    will be returned.

    Parameters
    ----------
    lat : ~astropy.units.Quantity, optional
        Geocentric (spherical) latitude. If given, a small latitude
        dependence will be considered.

    Returns
    -------
    l2 : ~astropy.units.Quantity
        Nominal degree 2 Shida number.

    Notes
    -----

    References
    ----------
    .. [1] IERS Conventions(2010), section 7.1.1, page 105.

    """
    if lat is not None:
        return l2 + 0.0002 * (3 * _np.sin(lat)**2 - 1) / 2
    else:
        return l2


def h2_love_number(lat: _u.deg = None) -> _u.dimensionless_unscaled:
    """Return degree 2 Love number (h2,0).

    If `lat` is None, the nominal degree 2 Love number h2=0.6078
    will be returned.

    Parameters
    ----------
    lat : ~astropy.units.Quantity, optional
        Geocentric (spherical) latitude. If given, a small latitude
        dependence will be considered.

    Returns
    -------
    h2 : ~astropy.units.Quantity
        Nominal degree 2 Love number.

    References
    ----------
    .. [1] IERS Conventions(2010), section 7.1.1, page 105.

    """
    if lat is not None:
        return h2 - 0.0006 * (3 * _np.sin(lat)**2 - 1) / 2
    else:
        return h2
