"""Extensions to `astropy.coordinates.representation`.

"""

import astropy.units as u
from astropy.coordinates.angles import Longitude, Latitude

from astropy.coordinates import BaseRepresentation, CartesianRepresentation

from pygeoid.coordinates import transform
from pygeoid.coordinates.ellipsoid import Ellipsoid


class GeodeticRepresentation(BaseRepresentation):

    attr_classes = {'lat': Latitude,
                    'lon': Longitude,
                    'height': u.Quantity}

    _ellipsoid = Ellipsoid()

    def __init__(self, lat, lon=None, height=None, ell=None, copy=True):
        if height is None and not isinstance(lat, self.__class__):
            height = 0 << u.m

        super().__init__(lat, lon, height, copy=copy)

        if ell is not None:
            self._ellipsoid = ell

        if not self.height.unit.is_equivalent(u.m):
            raise u.UnitTypeError(f"{self.__class__.__name__} requires "
                                  f"height with units of length.")

    @property
    def ellipsoid(self):
        """Reference ellipsoid.

        """
        return self._ellipsoid

    def to_cartesian(self):
        x, y, z = u.Quantity(
            transform.geodetic_to_cartesian(self.lat, self.lon,
                                            self.height, self._ellipsoid))
        return CartesianRepresentation(x, y, z, copy=False)

    @classmethod
    def from_cartesian(cls, cart, ell=None):
        x, y, z = cart.get_xyz()

        if ell is not None:
            cls._ellipsoid = ell

        lat, lon, height = transform.cartesian_to_geodetic(
            x, y, z, ell=cls._ellipsoid)
        return cls(lat, lon, height, copy=False)


class EllipsoidalHarmonicRepresentation(BaseRepresentation):

    attr_classes = {'rlat': Latitude,
                    'lon': Longitude,
                    'u_ax': u.Quantity}

    _ellipsoid = Ellipsoid()

    def __init__(self, rlat, lon=None, u_ax=None, ell=None, copy=True):

        super().__init__(rlat, lon, u_ax, copy=copy)

        if ell is not None:
            self._ellipsoid = ell

        if not self.u_ax.unit.is_equivalent(u.m):
            raise u.UnitTypeError(f"{self.__class__.__name__} requires "
                                  f"u_ax with units of length.")

    @property
    def ellipsoid(self):
        """Reference ellipsoid.

        """
        return self._ellipsoid

    def to_cartesian(self):
        x, y, z = u.Quantity(
            transform.ellipsoidal_to_cartesian(self.rlat, self.lon,
                                               self.u_ax, self._ellipsoid))
        return CartesianRepresentation(x, y, z, copy=False)

    @classmethod
    def from_cartesian(cls, cart, ell=None):
        x, y, z = cart.get_xyz()

        if ell is not None:
            cls._ellipsoid = ell

        rlat, lon, u_ax = transform.cartesian_to_ellipsoidal(
            x, y, z, ell=cls._ellipsoid)
        return cls(rlat, lon, u_ax, copy=False)
