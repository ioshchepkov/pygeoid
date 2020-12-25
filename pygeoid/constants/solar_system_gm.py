"""Standard gravitational parameters of the solar system bodies.

"""

from astropy.constants import Constant as _Constant

__all__ = ['gm_moon', 'gm_sun', 'gm_mars', 'gm_venus', 'gm_mercury',
           'gm_jupiter', 'gm_saturn', 'gm_uranus', 'gm_neptune']

# Import Moon and Sun
from pyshtools.constants.Moon import gm as gm_moon
from .iers2010 import GM_sun as gm_sun

# Import planets
from pyshtools.constants.Mars import gm as gm_mars
from pyshtools.constants.Venus import gm as gm_venus
from pyshtools.constants.Mercury import gm as gm_mercury

# Define planets
gm_jupiter = _Constant(
    abbrev='gm_jupiter',
    name='Gravitational constant times the mass of Jupiter',
    value=126686536.1e9,
    unit='m3 / s2',
    uncertainty=2.7e9,
    reference='Jacobson, R.A., (2013), JUP310 orbit solution.')

gm_saturn = _Constant(
    abbrev='gm_saturn',
    name='Gravitational constant times the mass of Saturn',
    value=37931208e9,
    unit='m3 / s2',
    uncertainty=1e9,
    reference='Jacobson, R. A., Antreasian, P. G., Bordi, J. J., '
    'Criddle, K. E., Ionasescu,R., Jones, J. B., Mackenzie, R. A., '
    'Pelletier, F. J., Owen Jr., W. M., Roth, D. C., and Stauch, J. R., '
    '(2006), The gravity field of the Saturnian system from satellite '
    'observations and spacecraft tracking data, '
    'Astronomical Journal 132, 6.')

gm_uranus = _Constant(
    abbrev='gm_uranus',
    name='Gravitational constant times the mass of Uranus',
    value=5793951.3e9,
    unit='m3 / s2',
    uncertainty=4.4e9,
    reference='Jacobson, R. A. (2014), The Orbits of the Uranian '
    'Satellites and Rings, the Gravity Field of the Uranian System, '
    'and the Orientation of the Pole of Uranus, '
    'Astronomical Journal 148, 76-88.')

gm_neptune = _Constant(
    abbrev='gm_neptune',
    name='Gravitational constant times the mass of Neptune',
    value=6835100e9,
    unit='m3 / s2',
    uncertainty=10e9,
    reference='Jacobson, R. A. (2009), The Orbits of the Neptunian '
    'Satellites and the Orientation of the Pole of Neptune, '
    'Astronomical Journal 137, 4322.')


_gm_body = {'moon' : gm_moon, 'sun' : gm_sun, 'mars' : gm_mars,
            'venus': gm_venus, 'mercury' : gm_mercury, 'jupiter': gm_jupiter,
            'saturn': gm_saturn, 'uranus': gm_uranus, 'neptune': gm_neptune}

bodies = tuple(_gm_body)


def get_body_gm(body):
    """Get standard gravitational parameter GM for a solar system body.

    Parameters
    ----------
    body : str
        The solar system body for which GM will be returned.

    Returns
    -------
    gm : `~astropy.constants.Constant`
        Standard gravitational parameter of the body.

    Notes
    -----
    One can check which bodies are available using:

        `pygeoid.constants.gm.bodies`

    """

    if body in _gm_body:
        return _gm_body[body]
    else:
        raise ValueError('No such body!')
