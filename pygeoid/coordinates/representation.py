"""Extensions to `astropy.coordinates.representation`.

"""

from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.coordinates import (BaseDifferential, BaseRepresentation,
                                 CartesianRepresentation)
from astropy.coordinates.angles import Latitude, Longitude
from pygeoid.coordinates import transform
from pygeoid.coordinates.ellipsoid import Ellipsoid


class GeodeticRepresentation(BaseRepresentation):

    attr_classes = OrderedDict([
        ('lon', Longitude),
        ('lat', Latitude),
        ('height', u.Quantity)])

    _ellipsoid = Ellipsoid()

    def __init__(self, lon, lat=None, height=None, ell=None, copy=True):
        if height is None and not isinstance(lat, self.__class__):
            height = 0 << u.m

        super().__init__(lon, lat, height, copy=copy)

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

    @property
    def lon(self):
        """
        The geodetic longitude of the point(s).

        """
        return self._lon

    @property
    def lat(self):
        """
        The geodetic latitude of the point(s).
        """
        return self._lat

    @property
    def height(self):
        """
        The geodetic height of the point(s).
        """
        return self._height

    def unit_vectors(self):
        sinlon, coslon = np.sin(self.lon), np.cos(self.lon)
        sinlat, coslat = np.sin(self.lat), np.cos(self.lat)
        return {
            'lon': CartesianRepresentation(
                -sinlon, coslon, 0., copy=False),
            'lat': CartesianRepresentation(-sinlat * coslon,
                                           -sinlat * sinlon, coslat, copy=False),
            'height': CartesianRepresentation(coslat * coslon,
                                              coslat * sinlon, sinlat, copy=False)
        }

    def scale_factors(self):
        pmer_rad = self._ellipsoid.meridian_curvature_radius(self.lat)
        sf_lat = (pmer_rad + self.height) / u.radian
        pver_rad = self._ellipsoid.prime_vertical_curvature_radius(self.lat)
        sf_lon = (pver_rad + self.height) * np.cos(self.lat) / u.radian
        sf_height = np.broadcast_to(1. * u.one, self.shape, subok=True)
        return {'lon': sf_lon,
                'lat': sf_lat,
                'height': sf_height}

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
        return GeodeticRepresentation(lon=lon, lat=lat, height=height,
                ell=cls._ellipsoid, copy=True)


class GeodeticDifferential(BaseDifferential):
    base_representation = GeodeticRepresentation


class EllipsoidalHarmonicRepresentation(BaseRepresentation):

    attr_classes = OrderedDict([
        ('rlat', Latitude),
        ('lon', Longitude),
        ('u_ax', u.Quantity)])

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

    def unit_vectors(self):
        sinlon, coslon = np.sin(self.lon), np.cos(self.lon)
        sinrlat, cosrlat = np.sin(self.rlat), np.cos(self.rlat)
        le2 = self._ellipsoid.linear_eccentricity**2
        u_ax2 = self.u_ax**2
        k = np.sqrt(u_ax2 + le2)

        w = np.sqrt(u_ax2 + le2 * sinrlat) / k
        uwk = self.u_ax / (w * k)

        uv_u_ax = (
            uwk * cosrlat * coslon,
            uwk * cosrlat * sinlon,
            sinrlat / w)

        uv_rlat = (
            -sinrlat * coslon / w,
            -sinrlat * sinlon / w,
            uwk * cosrlat,
        )

        uv_lon = (-sinlon, coslon, 0)

        return {
            'lon': CartesianRepresentation(*uv_lon, copy=False),
            'rlat': CartesianRepresentation(*uv_rlat, copy=False),
            'u_ax': CartesianRepresentation(*uv_u_ax, copy=False)
        }

    def scale_factors(self):
        le2 = self._ellipsoid.linear_eccentricity**2
        u_ax2 = self.u_ax**2
        k = np.sqrt(u_ax2 + le2)

        sf_rlat = np.sqrt(u_ax2 + le2 * np.sin(self.rlat)**2) / u.radian
        sf_u_ax = sf_rlat / k
        sf_lon = k * np.cos(self.rlat) / u.radian

        return {'lon': sf_lon,
                'rlat': sf_rlat,
                'u_ax': sf_u_ax}

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


class EllipsoidalHarmonicDifferential(BaseDifferential):
    base_representation = EllipsoidalHarmonicRepresentation
