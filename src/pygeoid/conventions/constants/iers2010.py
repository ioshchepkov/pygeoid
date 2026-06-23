"""IERS Conventions (2010) numerical standards.

These constants reproduce the **updated** values from IERS Conventions 2010
v1.3.0.
See the IERS Conventions Center:
https://iers-conventions.obspm.fr/
"""

import numpy as _np
from astropy.constants import Constant as _Constant

from pygeoid.conventions import units as _u

####################################################
# IERS2010 Conventions
####################################################


class IERS2010(_Constant):
    default_reference = "IERS Conventions (2010)"
    _registry = {}
    _has_incompatible_units = set()


####################################################
# Natural defining constants
####################################################

c = IERS2010(
    abbrev="c",
    name="Speed of light (SI defining constant)",
    value=299792458.0,
    unit="m / s",
    uncertainty=0,
    system="si",
)

####################################################
# Auxiliary defining constants
####################################################

k = IERS2010(
    abbrev="k",
    name="Gaussian gravitational constant (scale-independent)",
    value=1.720209895e-2,
    unit="",
    uncertainty=0,
    system="si",
)

L_G = IERS2010(
    abbrev="L_G",
    name="1 - d(TT)/d(TCG) (TT/TCG rate constant)",
    value=6.969290134e-10,
    unit="",
    uncertainty=0,
    system="si",
)

L_B = IERS2010(
    abbrev="L_B",
    name="1 - d(TDB)/d(TCB) (TDB/TCB rate constant)",
    value=1.550519768e-8,
    unit="",
    uncertainty=0,
    system="si",
)

TDB0 = IERS2010(
    abbrev="TDB0",
    name="TDB - TCB at JD 2443144.5 TAI (TDB/TCB offset)",
    value=-6.55e-5,
    unit="s",
    uncertainty=0,
    system="si",
)

theta0 = IERS2010(
    abbrev="theta0",
    name="Earth Rotation Angle at J2000.0 (ERA/UT1)",
    value=0.7790572732640,
    unit=_u.rev,
    uncertainty=0,
    system="si",
)

dtheta_dt = IERS2010(
    abbrev="dtheta_dt",
    name="Rate of advance of Earth Rotation Angle (UT1 day)",
    value=1.00273781191135448,
    unit=_u.rev / _u.day,
    uncertainty=0,
    system="si",
)

####################################################
# Natural measurable constants
####################################################

G = IERS2010(
    abbrev="G",
    name="Constant of gravitation (SI/proper scale)",
    value=6.67428e-11,
    unit="m**3 / (kg * s**2)",
    uncertainty=6.7e-15,
    system="si",
)

####################################################
# Body constants
####################################################

GM_sun = IERS2010(
    abbrev="GM_sun",
    name="Heliocentric gravitational constant (TCB-compatible)",
    value=1.32712442099e20,
    unit="m**3 / s**2",
    uncertainty=1e10,
    system="si",
)

J2_sun = IERS2010(
    abbrev="J2_sun",
    name="Dynamical form factor of the Sun (scale-independent)",
    value=2.0e-7,
    unit="",
    uncertainty=_np.nan,
    system="si",
)

mu = IERS2010(
    abbrev="mu",
    name="Moon-Earth mass ratio (scale-independent)",
    value=0.0123000371,
    unit="",
    uncertainty=4.0e-10,
    system="si",
)

####################################################
# Earth constants
####################################################

GM_earth_tcg = IERS2010(
    abbrev="GM_earth",
    name="Geocentric gravitational constant (TCG-compatible)",
    value=3.986004418e14,
    unit="m**3 / s**2",
    uncertainty=8e15,
    system="si",
)

GM_earth_tt = IERS2010(
    abbrev="GM_earth",
    name="Geocentric gravitational constant (TT-compatible)",
    value=3.986004415e14,
    unit="m**3 / s**2",
    uncertainty=8e15,
    system="si",
)

a = IERS2010(
    abbrev="a",
    name="Equatorial radius of the Earth (TCG-compatible, zero tide)",
    value=6378136.6,
    unit="m",
    uncertainty=0.1,
    system="si",
)

J2_earth = IERS2010(
    abbrev="J2_earth",
    name="Dynamical form factor of the Earth (scale-independent, zero tide)",
    value=0.0010826359,
    unit="",
    uncertainty=1e-10,
    system="si",
)

rf = IERS2010(
    abbrev="rf",
    name="Inverse flattening factor of the Earth (scale-independent, zero tide)",
    value=298.25642,
    unit="",
    uncertainty=0.00001,
    system="si",
)

ge = IERS2010(
    abbrev="ge",
    name="Mean equatorial gravity (TCG-compatible, zero tide)",
    value=9.7803278,
    unit="m / s**2",
    uncertainty=0.00001,
    system="si",
)

W0 = IERS2010(
    abbrev="W0",
    name=(
        "Potential of the geoid (TCG-compatible, zero tide; same as the IHRF "
        "conventional W0 value from IAG 2015 Resolution No. 1)"
    ),
    value=62636853.4,
    unit="m**2 / s**2",
    uncertainty=0.02,
    system="si",
)

W0_IHRF = W0

R0 = IERS2010(
    abbrev="R0",
    name="Geopotential scale factor GM_earth / W0 (TCG-compatible)",
    value=6363672.6,
    unit="m",
    uncertainty=0.1,
    system="si",
)

H = IERS2010(
    abbrev="H",
    name="Dynamical flattening (scale-independent)",
    value=3273795e-9,
    unit="",
    uncertainty=1e-9,
    system="si",
)


####################################################
# Initial value at J2000.0
####################################################

eps0 = IERS2010(
    abbrev="eps0",
    name="Obliquity of the ecliptic at J2000.0 (scale-independent)",
    value=84381.40600,
    unit="arcsec",
    uncertainty=0.00100,
    system="si",
)


####################################################
# Other constants
####################################################

au = IERS2010(
    abbrev="au",
    name="Astronomical unit (TDB-compatible)",
    value=1.49597870700e11,
    unit="m",
    uncertainty=3.0,
    system="si",
)

L_C = IERS2010(
    abbrev="L_C",
    name="Average value of 1 - d(TCG)/d(TCB) (TCG/TCB rate constant)",
    value=1.48082686741e-8,
    unit="",
    uncertainty=2.0e-17,
    system="si",
)


def tcg_to_tt(x):
    """Convert TCG-compatible value to TT-compatible value."""
    return x * (1 - L_G)


####################################################
# Love and Shida numbers of the second-degree
####################################################


k2 = IERS2010(
    abbrev="k2",
    name="Nominal degree 2 Love number k2",
    value=0.29525,
    unit="",
    uncertainty=0,
    system="si",
)

h2 = IERS2010(
    abbrev="h2",
    name="Nominal degree 2 Love number h2",
    value=0.6078,
    unit="",
    uncertainty=0,
    system="si",
)

l2 = IERS2010(
    abbrev="l2",
    name="Nominal degree 2 Shida number l2",
    value=0.0847,
    unit="",
    uncertainty=0,
    system="si",
)

DEGREE2_LOVE_NUMBERS = {"k": k2, "l": l2, "h": h2}

k20 = IERS2010(
    abbrev="k20",
    name="Nominal degree 2 Love number k20",
    value=0.30190,
    unit="",
    uncertainty=0,
    system="si",
)


def l2_shida_number(lat: _u.deg = None) -> _u.dimensionless_unscaled:
    """Return degree 2 Shida number (l2,0).

    If `lat` is None, the nominal degree 2 Shida number l2=0.0847
    will be returned.

    Parameters
    ----------
    lat : ~pygeoid.conventions.units.Quantity, optional
        Geocentric (spherical) latitude. If given, a small latitude
        dependence will be considered.

    Returns
    -------
    l2 : ~pygeoid.conventions.units.Quantity
        Nominal degree 2 Shida number.

    Notes
    -----

    References
    ----------
    - IERS Conventions (2010), section 7.1.1, page 105.

    """
    if lat is not None:
        return l2 + 0.0002 * (3 * _np.sin(lat) ** 2 - 1) / 2
    else:
        return l2


def h2_love_number(lat: _u.deg = None) -> _u.dimensionless_unscaled:
    """Return degree 2 Love number (h2,0).

    If `lat` is None, the nominal degree 2 Love number h2=0.6078
    will be returned.

    Parameters
    ----------
    lat : ~pygeoid.conventions.units.Quantity, optional
        Geocentric (spherical) latitude. If given, a small latitude
        dependence will be considered.

    Returns
    -------
    h2 : ~pygeoid.conventions.units.Quantity
        Nominal degree 2 Love number.

    References
    ----------
    - IERS Conventions (2010), section 7.1.1, page 105.

    """
    if lat is not None:
        return h2 - 0.0006 * (3 * _np.sin(lat) ** 2 - 1) / 2
    else:
        return h2
