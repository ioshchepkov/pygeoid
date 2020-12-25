"""Collection of useful constants.

"""

from numpy import pi
import astropy.units as u
from astropy.constants import Constant as _Constant

from .solar_system_gm import get_body_gm

####################################################
# Import constants
####################################################

from astropy.constants import G
from astropy.constants import g0
from astropy.constants import R_earth
from .iers2010 import GM_earth_tt as gm_earth

from .iers2010 import DEGREE2_LOVE_NUMBERS

####################################################
# Define constants
####################################################

W0_IHRF = _Constant(
    abbrev='W0',
    name='Potential of the geoid for the International Height Reference Frame(IHRF)',
    value=62636853.4,
    unit='m**2 / s**2',
    uncertainty=0.02,
    reference='IAG 2015 Resolution No.1')

####################################################
# Frequently used expressions
####################################################

# n*pi
_2pi = 2 * pi
_4pi = 4 * pi

# 2*pi*G
_2piG = _2pi * G
