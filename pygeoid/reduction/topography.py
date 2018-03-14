"""Topographic reduction in gravity and geoid modelling"""

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
    height = height*u.m
    density = density*(u.kg / u.m**3)
    return (_2piG*density*height).to(units).magnitude
