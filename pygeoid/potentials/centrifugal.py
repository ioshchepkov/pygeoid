
import astropy.units as u
import numpy as np
from pygeoid.reductions.core import PotentialBase

from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import CartesianDifferential


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

    def _potential(self, position):
        rep = position.represent_as(CartesianRepresentation)
        return 0.5 * self.omega**2 * (rep.x**2 + rep.y**2)

    def _differential(self, position):
        rep = position.represent_as(CartesianRepresentation)
        return CartesianDifferential(
            self.omega**2 * rep.x,
            self.omega**2 * rep.y,
            np.zeros(position.shape) * u.m / u.s**2)
