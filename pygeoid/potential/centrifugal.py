
import numpy as np
import astropy.units as u

from pygeoid.potential.core import PotentialBase


class Centrifugal(PotentialBase):
    """Centrifugal potential and its derivatives.

    Parameters
    ----------
    omega : ~astropy.units.Quantity
        Angular rotation rate of the body, in 1 / s.
        Default value is the angular speed of
        Earth's rotation 7292115e-11 1/s
    """

    @u.quantity_input
    def __init__(self, omega: u.s**-1 = 7292115e-11 / u.s):
        self._omega = omega

    @property
    def omega(self):
        return self._omega

    @u.quantity_input
    def _potential(self, position) -> u.m**2 / u.s**2:
        """Return centrifugal potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        return 0.5 * self.omega**2 * (position.x**2 + position.y**2)

    def _derivative_spherical(self, position, variable):
        sph = position.represent_as('spherical')
        if variable in ('lat', 'latitude'):
            cslat = np.cos(sph.lat) * np.sin(sph.lat)
            return -self.omega**2 * sph.distance**2 * cslat / u.radian
        elif variable in ('lon', 'longitude', 'long'):
            return np.zeros(position.shape) * u.m**2 / u.s**2 / u.radian
        elif variable in ('distance', 'radius', 'r', 'radial'):
            return self.omega**2 * sph.distance * np.cos(sph.lat)**2
        else:
            raise ValueError('No variable named {0}'.format(variable))

    def _derivative_cartesian(self, position, variable):
        if variable == 'x':
            return self.omega**2 * position.x
        elif variable == 'y':
            return self.omega**2 * position.y
        elif variable == 'z':
            return np.zeros(position.shape) * u.m / u.s**2
        else:
            raise ValueError('No variable named {0}'.format(variable))
