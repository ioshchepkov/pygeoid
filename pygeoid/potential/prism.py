"""Gravitational field of a prism.

"""

import itertools
import numpy as np
from typing import Annotated, NamedTuple

from pygeoid.constants import G
from pygeoid.coordinates.frame import LocalTangentPlane
from pygeoid.potential.core import PotentialBase as _PotentialBase
import astropy.units as u

class ForwardModel:

    @property
    def density(self):
        """Return density value.

        """
        return self._density

    def gravitation(self, x, y, z):
        """Return gravitation value.

        """
        return np.sqrt(self.gx(x, y, z)**2 + self.gy(x, y, z)**2 +
                       self.gz(x, y, z)**2)

    def tensor(self, x, y, z):
        """Return gradient tensor.

        """
        gxx = self.gxx(x, y, z)
        gyy = self.gyy(x, y, z)
        gzz = self.gzz(x, y, z)
        gxy = self.gxy(x, y, z)
        gxz = self.gxy(x, y, z)
        gyz = self.gxy(x, y, z)

        tensor = np.array([
            [gxx, gxy, gxz],
            [gxy, gyy, gyz],
            [gxz, gyz, gzz]])

        return tensor

    def invariants(self, x, y, z):
        """Return invariants of the gradient tensor.

        """
        tensor = self.tensor(x, y, z)
        i_1 = np.trace(tensor)
        i_2 = 0.5 * (np.trace(tensor)**2 - np.trace(tensor**2))
        i_3 = np.linalg.det(tensor)
        return i_1, i_2, i_3


def _limits_sum(function):
    """Sum function by rectangular limits.

    """

    #def wraper(self, x, y, z):
    def wraper(self, position):

        x1b, x2b, y1b, y2b, z1b, z2b = self._bounds
        cond = (x >= x1b) & (x <= x2b) & (y >= y1b) & (
            y <= y2b) & (z >= z1b) & (z <= z2b)

        if np.any(cond):
            raise ValueError('Point within or on the prism!')

        x1 = self._bounds[0] - x
        x2 = self._bounds[1] - x
        y1 = self._bounds[2] - y
        y2 = self._bounds[3] - y
        z1 = self._bounds[4] - z
        z2 = self._bounds[5] - z
        bounds = (x1, x2, y1, y2, z1, z2)

        total_sum = 0
        for index in itertools.product([1, 2], [3, 4], [5, 6], repeat=1):
            index = np.asarray(index)
            coords = np.asarray(bounds)[index - 1]
            total_sum += (-1)**(index.sum()) * function(self, *coords)
        return total_sum * G * self.density
    return wraper


def _radius(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


class PrismBounds(NamedTuple):
    west : u.Quantity[u.m]
    east : u.Quantity[u.m]
    south : u.Quantity[u.m]
    north : u.Quantity[u.m]
    top : u.Quantity[u.m]
    bottom : u.Quantity[u.m]


class Prism(ForwardModel):
    """External gravitational field of the right rectangular prism.

    x -> East, y -> North, z -> Down

    Parameters
    ----------
    bounds : (x1, x2, y1, y2, z1, z2), floats
        West, east, south, north, top and bottom of the prism, in metres.
    density : float
        Density of the prism, in kg/m**3.

    Notes
    -----
    The formulas from Nagy et al.[1]_ are used in this class.

    References
    ----------
    .. [1] Nagy, D., Papp, G. and Benedek, J., 2000. The gravitational potential
    and its derivatives for the prism. Journal of Geodesy, 74(7-8), pp.552-560.
    """

    @u.quantity_input
    def __init__(self, bounds: PrismBounds,
            density:u.kg/u.m**3=1.0 * u.kg / u.m**3):

        self._bounds = PrismBounds(*bounds)
        self._density = density
        self._frame = LocalTangentPlane(orientation=('E', 'N', 'D'))

    @_limits_sum
    def potential(self, position):
        """Calculate the gravitational potential V of the prism.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        r = _radius(x, y, z)
        out = x * y * np.log(z + r) + y * z * \
            np.log(x + r) + z * x * np.log(y + r)
        out -= 0.5 * x**2 * np.arctan2(y * z, x * r)
        out -= 0.5 * y**2 * np.arctan2(z * x, y * r)
        out -= 0.5 * z**2 * np.arctan2(x * y, z * r)
        return out

    @_limits_sum
    def gx(self, x, y, z):
        """Calculate the first derivative of the potential Vx = gx.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        r = _radius(x, y, z)
        out = y * np.log(z + r)
        out += z * np.log(y + r)
        out -= x * np.arctan2(y * z, x * r)
        return -out

    @_limits_sum
    def gy(self, x, y, z):
        """Calculate the first derivative of the potential Vy = gy.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        r = _radius(x, y, z)
        out = z * np.log(x + r)
        out += x * np.log(z + r)
        out -= y * np.arctan2(z * x, y * r)
        return -out

    @_limits_sum
    def gz(self, x, y, z):
        """Calculate the first derivative of the potential Vz = gz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        r = _radius(x, y, z)
        out = x * np.log(y + r)
        out += y * np.log(x + r)
        out -= z * np.arctan2(x * y, z * r)
        return -out

    @_limits_sum
    def gxx(self, x, y, z):
        """Calculate the second derivative of the potential Vxx = gxx.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        return np.arctan2(y * z, x * _radius(x, y, z))

    @_limits_sum
    def gyy(self, x, y, z):
        """Calculate the second derivative of the potential Vyy = gyy.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        return np.arctan2(z * x, y * _radius(x, y, z))

    @_limits_sum
    def gzz(self, x, y, z):
        """Calculate the second derivative of the potential Vzz = gzz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        return np.arctan2(x * y, z * _radius(x, y, z))

    @_limits_sum
    def gxz(self, x, y, z):
        """Calculate the second derivative of the potential Vxz = gxz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        return -np.log(y + _radius(x, y, z))

    @_limits_sum
    def gyz(self, x, y, z):
        """Calculate the second derivative of the potential Vyz = gyz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        return -np.log(x + _radius(x, y, z))

    @_limits_sum
    def gxy(self, x, y, z):
        """Calculate the second derivative of the potential Vxy = gxy.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        x, y, z : float or array_like of floats
            Cartesian coordinates of the attracted point, in metres.
        """
        return -np.log(z + _radius(x, y, z))
