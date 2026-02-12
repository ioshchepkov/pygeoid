"""Collection of useful constants.

"""

from astropy.constants import Constant as _Constant

####################################################
# Import constants
####################################################
from astropy.constants import G, g0
from numpy import pi

from .iers2010 import DEGREE2_LOVE_NUMBERS as DEGREE2_LOVE_NUMBERS
from .iers2010 import GM_earth_tt as GM_earth_tt
from .solar_system_gm import get_body_gm as get_body_gm

gm_earth = GM_earth_tt
standart_gravity_acceleration = g0
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
