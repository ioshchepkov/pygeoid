"""
Collection of the constants for the gravity field modelling
"""

from scipy.constants import G, pi
from pygeoid.units import units as u


# Constant of gravitation (CODATA2014)
G = G * u.m**3 / (u.kg * u.s**2)

# Potential of the geoid for the Iternational Height Reference Frame (IHRF)
W0_IHRF = 62636853.4 * u.m**2 / u.s**2

####################################################
# IERS2010 Convention
####################################################

# Constant of gravitation
G_IERS2010 = 6.67428e-11 * u.m**3 / (u.kg * u.s**2)
# Potential of the geoid
W0_IERS2010 = 62636856.0 * u.m**2 / u.s**2
# Geocentric gravitational constant
GM_IERS2010 = 3.986004418e14 * u.m**3 / u.s**2
# Equatorial radius of the Earth
a_IERS2010 = 6378136.6 * u.m
# Dynamical form factor of the Earth
J2_IERS2010 = 1.0826359e-3
# Flattening factor of the Earth
rf_IERS2010 = 298.25642
# Mean equatorial gravity
ge_IERS2010 = 9.7803278 * u.m / u.s**2

####################################################
# Frequently used expressions
####################################################

# n*pi
_2pi = 2 * pi
_4pi = 4 * pi

# 2*pi*G
_2piG = _2pi * G
