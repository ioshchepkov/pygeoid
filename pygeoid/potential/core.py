
import abc

import numpy as np
import astropy.units as u


class PotentialBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _potential(self, position):
        pass

    @abc.abstractmethod
    def _gradient(self, position):
        pass

    @abc.abstractmethod
    def _gradient_vector(self, position):
        pass

    @abc.abstractmethod
    def _derivative(self, position, *args, **kwargs):
        pass

    def potential(self, position):
        return self._potential(position=position)

    def gradient(self, position):
        return self._gradient(position=position)

    def gradient_vector(self, position, coordinates=None):
        return self._gradient_vector(position=position,
                coordinates=coordinates)

    def derivative(self, position, variable, coordinates=None):
        return self._derivative(position=position, variable=variable,
                coordinates=coordinates)

class Centrifugal(PotentialBase):
    """Centrifugal potential and its derivatives.

    Parameters
    ----------
    omega : float
        Angular rotation rate of the body, in rad/s.
        Default value is the angular speed of
        Earth's rotation 7292115e-11 rad/s
    """

    @u.quantity_input
    def __init__(self, omega=7292115e-11  / u.s):

        self.omega = omega

    @u.quantity_input
    def _potential(self, position) -> u.m**2 / u.s**2:
        """Return centrifugal potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        return 0.5 * self.omega**2 * (position.x**2 + position.y**2)

    @u.quantity_input
    def _gradient(self, position) -> u.m / u.s**2:
        """Return centrifugal force.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        qx = self.derivative(position, 'x', 'cartesian')
        qy = self.derivative(position, 'y', 'cartesian')

        return np.sqrt(qx**2 + qy**2)

    def _gradient_vector(self, position, coordinates=None):

        if coordinates is None:
            coordinates = position.representation_type.get_name()

        if coordinates == 'spherical':
            q1 = self.derivative(position, 'lat', 'spherical')
            q2 = self.derivative(position, 'lon', 'spherical')
            q3 = self.derivative(position, 'radius', 'spherical')
        elif coordinates == 'cartesian':
            q1 = self.derivative(position, 'x', 'cartesian')
            q2 = self.derivative(position, 'y', 'cartesian')
            q3 = self.derivative(position, 'z', 'cartesian')

        return (q1, q2, q3)

    def _derivative(self, position, variable='lat', coordinates=None):

        if coordinates is None:
            coordinates = position.representation_type.get_name()

        if coordinates == 'spherical':
            sph = position.represent_as('spherical')
            if variable in ('lat', 'latittude'):
                cslat = np.cos(sph.lat) * np.sin(sph.lat)
                return -self.omega**2 * sph.distance**2 * cslat
            elif variable in ('lon', 'longitude'):
                return np.zeros(position.shape) * u.m**2 / u.s**2
            elif variable in ('distance', 'radius', 'r'):
                return self.omega**2 * sph.distance * np.cos(sph.lat)**2
        elif coordinates == 'cartesian':
            if variable == 'x':
                return self.omega**2 * position.x
            if variable == 'y':
                return self.omega**2 * position.y
            if variable == 'z':
                return np.zeros(position.shape) * u.m / u.s**2

    def hessian(self, position, coordinates='spherical'):
        pass
