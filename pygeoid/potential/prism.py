"""Gravitational field of a prism.

"""

import itertools
from typing import NamedTuple

import astropy.units as u
import numpy as np
from pygeoid.constants import G
from pygeoid.coordinates.frame import LocalFrame
from pygeoid.potential.core import PotentialBase as _PotentialBase


def _limits_sum(function):
    """Sum function by rectangular limits.

    """

    def wraper(self, position):

        x1b, x2b, y1b, y2b, z1b, z2b = self._bounds
        x, y, z = position.cartesian.get_xyz()

        cond = (x >= x1b) & (x <= x2b) & (y >= y1b) & (
            y <= y2b) & (z >= z1b) & (z <= z2b)

        if np.any(cond):
            raise ValueError('Point within or on the prism!')

        x1 = x1b - x
        x2 = x2b - x
        y1 = y1b - y
        y2 = y2b - y
        z1 = z1b - z
        z2 = z2b - z

        bounds = u.Quantity([x1, x2, y1, y2, z1, z2])

        total_sum = 0
        for index in itertools.product([1, 2], [3, 4], [5, 6], repeat=1):
            index = np.asarray(index)
            coords = LocalFrame(bounds[index - 1])
            total_sum += (-1)**(index.sum()) * function(self, coords)
        return total_sum * G * self._density
    return wraper


class PrismBounds(NamedTuple):
    west : u.Quantity[u.m]
    east : u.Quantity[u.m]
    south : u.Quantity[u.m]
    north : u.Quantity[u.m]
    top : u.Quantity[u.m]
    bottom : u.Quantity[u.m]


class Prism(_PotentialBase):
    """External gravitational field of the right rectangular prism.

    x -> East, y -> North, z -> Down

    Parameters
    ----------
    bounds : (x1, x2, y1, y2, z1, z2), ~astropy.units.Quantity
        West, east, south, north, top and bottom of the prism.
    density : ~astropy.units.Quantity
        Density of the prism.

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
                 density: u.kg / u.m**3 = 1.0 * u.kg / u.m**3):

        self._bounds = PrismBounds(*bounds)
        self._density = density

    @staticmethod
    def _prepare_coords(position):
        x, y, z = position.cartesian.get_xyz().value
        r = position.cartesian.norm().value
        return x, y, z, r

    @_limits_sum
    def _potential(self, position):
        """Calculate the gravitational potential V of the prism.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)

        out = x * y * np.log(z + r) + y * z * \
            np.log(x + r) + z * x * np.log(y + r)
        out -= 0.5 * x**2 * np.arctan2(y * z, x * r)
        out -= 0.5 * y**2 * np.arctan2(z * x, y * r)
        out -= 0.5 * z**2 * np.arctan2(x * y, z * r)

        return out * u.m**2

    def _derivative_cartesian(self, position, variable):
        if variable == 'x':
            return self.gx(position)
        elif variable == 'y':
            return self.gy(position)
        elif variable == 'z':
            return self.gz(position)
        else:
            raise ValueError('No variable named {0}'.format(variable))

    def _hessian(self, position):
        """Return gradient tensor.

        """

        gxx = self.gxx(position).value
        gyy = self.gyy(position).value
        gzz = self.gzz(position).value
        gxy = self.gxy(position).value
        gxz = self.gxy(position).value
        gyz = self.gxy(position).value

        tensor = np.asarray([
            [gxx, gxy, gxz],
            [gxy, gyy, gyz],
            [gxz, gyz, gzz]
        ])

        return u.Quantity(tensor.T, 1 / u.s**2)

    def invariants(self, position):
        """Return invariants of the gradient tensor.

        """
        tensor = self._hessian(position)
        i_1 = np.trace(tensor, axis1=-1, axis2=-2)
        i_2 = 0.5 * (np.trace(tensor, axis1=-1)**2 -
                     np.trace(tensor**2, axis1=-1))
        i_3 = np.linalg.det(tensor)
        return i_1, i_2, i_3

    @_limits_sum
    def gx(self, position):
        """Calculate the first derivative of the potential Vx = gx.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)

        out = y * np.log(z + r)
        out += z * np.log(y + r)
        out -= x * np.arctan2(y * z, x * r)
        return -out * u.m

    @_limits_sum
    def gy(self, position):
        """Calculate the first derivative of the potential Vy = gy.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)

        out = z * np.log(x + r)
        out += x * np.log(z + r)
        out -= y * np.arctan2(z * x, y * r)
        return -out * u.m

    @_limits_sum
    def gz(self, position):
        """Calculate the first derivative of the potential Vz = gz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)

        out = x * np.log(y + r)
        out += y * np.log(x + r)
        out -= z * np.arctan2(x * y, z * r)
        return -out * u.m

    @_limits_sum
    def gxx(self, position):
        """Calculate the second derivative of the potential Vxx = gxx.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)
        return np.arctan2(y * z, x * r)

    @_limits_sum
    def gyy(self, position):
        """Calculate the second derivative of the potential Vyy = gyy.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)
        return np.arctan2(z * x, y * r)

    @_limits_sum
    def gzz(self, position):
        """Calculate the second derivative of the potential Vzz = gzz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)
        return np.arctan2(x * y, z * r)

    @_limits_sum
    def gxz(self, position):
        """Calculate the second derivative of the potential Vxz = gxz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)
        return -np.log(y + r)

    @_limits_sum
    def gyz(self, position):
        """Calculate the second derivative of the potential Vyz = gyz.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)
        return -np.log(x + r)

    @_limits_sum
    def gxy(self, position):
        """Calculate the second derivative of the potential Vxy = gxy.

        x -> East, y -> North, z -> Down

        Parameters
        ----------
        position : ~pygeoid.coordinates.LocalFrame
            Cartesian coordinates of the attracted point.
        """
        x, y, z, r = self._prepare_coords(position)
        return -np.log(z + r)
