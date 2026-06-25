"""Position classes."""

import inspect

from astropy.coordinates import BaseCoordinateFrame
from astropy.coordinates.angles import Latitude, Longitude
from astropy.coordinates.representation import CartesianRepresentation

from pygeoid.conventions import units as u
from pygeoid.geometry import transform
from pygeoid.geometry.ellipsoid import DEFAULT_ELLIPSOID, Ellipsoid
from pygeoid.geometry.representation import (
    EllipsoidalHarmonicRepresentation,
    GeodeticRepresentation,
)

__all__ = ["Position"]


class Position(BaseCoordinateFrame):
    """Earth-Centered, Earth-Fixed frame.

    Parameters
    ----------
    ell : instance of the `pygeoid.geometry.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.
    *args
        Any representation of the frame data, e.g. x, y, and z coordinates.
    **kwargs
        Any extra BaseCoordinateFrame arguments.

    """

    default_representation = CartesianRepresentation
    """Default representation of local frames"""

    def __init__(self, *args, ell=None, **kwargs):

        super().__init__(*args, **kwargs)

        if ell is None:
            ell = Ellipsoid(DEFAULT_ELLIPSOID)
        self._ellipsoid = ell

    @property
    def ellipsoid(self):
        """Reference ellipsoid."""
        return self._ellipsoid

    @ellipsoid.setter
    def ellipsoid(self, ellipsoid):
        if not isinstance(ellipsoid, Ellipsoid):
            raise ValueError(
                "elliposid should be an instance of the "
                "`pygeoid.geometry.ellipsoid.Ellipsoid`!"
            )
        else:
            self._ellipsoid = ellipsoid

    @classmethod
    def from_spherical(cls, lat, lon, radius):
        """Position, initialized from spherical coordinates.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity or array-like
            Spherical latitude. Can be anything that initialises an
            `~astropy.coordinates.Latitude` object.
            (if array-like, in degrees).
        lon : ~pygeoid.conventions.units.Quantity or array-like
            Spherical longitude. Can be anything that initialises an
            `~astropy.coordinates.Longitude` object.
            (if array-like, in degrees).
        radius : ~pygeoid.conventions.units.Quantity or array-like
            Radius (if array-like, in metres).
        """
        lat = Latitude(lat, u.degree, copy=False)
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)

        if not isinstance(radius, u.Quantity):
            radius = u.Quantity(radius, u.m, copy=False)

        x, y, z = u.Quantity(transform.spherical_to_cartesian(lat, lon, radius))

        return cls(x, y, z)

    @classmethod
    def from_geodetic(cls, lat, lon, height=0.0, ell=None):
        """Position, initialized from geodetic coordinates.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity or array-like
            Geodetic latitude. Can be anything that initialises an
            `~astropy.coordinates.Latitude` object (if array-like, in degrees).
        lon : ~pygeoid.conventions.units.Quantity or array-like
            Geodetic longitude. Can be anything that initialises an
            `~astropy.coordinates.Longitude` object (if array-like, in degrees).
        height : ~pygeoid.conventions.units.Quantity or array-like
            Geodetic height (if array-like, in metres). Default is 0 m.
        ell : ~`pygeoid.geometry.ellipsoid.Ellipsoid`, optional
            Reference ellipsoid to which geodetic coordinates are referenced to.
            Default is None, which means the default ellipsoid of the class
            instance, but if given, it also will change the ellipsoid for
            the class instance.

        """
        lat = Latitude(lat, u.degree, copy=False)
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)

        if not isinstance(height, u.Quantity):
            height = u.Quantity(height, u.m, copy=False)

        if ell is None:
            ell = Ellipsoid(DEFAULT_ELLIPSOID)

        x, y, z = u.Quantity(transform.geodetic_to_cartesian(lat, lon, height, ell))

        self = cls(x, y, z)
        self._ellipsoid = ell

        return self

    @property
    def geodetic(self):
        return GeodeticRepresentation.from_cartesian(
            self.cartesian, ell=self._ellipsoid
        )

    @classmethod
    def from_ellipsoidal_harmonic(cls, rlat, lon, u_ax, ell=None):
        """Position, initialized from ellipsoidal-harmonic coordinates.

        Parameters
        ----------
        rlat : ~pygeoid.conventions.units.Quantity or array-like
            Reduced latitude. Can be anything that initialises an
            `~astropy.coordinates.Latitude` object.
        lon : ~pygeoid.conventions.units.Quantity or array-like
            Spherical longitude. Can be anything that initialises an
            `~astropy.coordinates.Longitude` object.
            (if array-like, in degrees).
        u_ax : ~pygeoid.conventions.units.Quantity or array-like
            Polar axis of the ellipsoid passing through the given point
            (if array-like, in metres).
        ell : ~`pygeoid.geometry.ellipsoid.Ellipsoid`
            Reference ellipsoid to which coordinates are referenced to.
            Default is None, which means the default ellipsoid of the class
            instance, but if given, it also will change the ellipsoid for
            the class instance.
        """
        rlat = Latitude(rlat, u.degree, copy=False)
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)

        if not isinstance(u_ax, u.Quantity):
            u_ax = u.Quantity(u_ax, u.m, copy=False)

        if ell is None:
            ell = Ellipsoid(DEFAULT_ELLIPSOID)

        x, y, z = u.Quantity(
            transform.ellipsoidal_to_cartesian(rlat, lon, u_ax, ell=ell)
        )

        self = cls(x, y, z)
        self._ellipsoid = ell

        return self

    @property
    def ellipsoidal_harmonic(self):
        return EllipsoidalHarmonicRepresentation.from_cartesian(
            self.cartesian, ell=self._ellipsoid
        )

    @u.quantity_input
    def enu(self, origin: tuple[u.deg, u.deg, u.m], ell=None):
        """Return local east-north-up cartesian coordinates.

        Parameters
        ----------
        origin : tuple of ~pygeoid.conventions.units.Quantity
            Ggeocentric (spherical) or geodetic coordinates of the origin
            (`lat0`, `lon0`, `r0`) or (`lat0`, `lon0`, `h0`).
        ell : instance of the `pygeoid.geometry.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates
            are referenced to. Default is None, meaning spherical
            coordinates instead of geodetic.

        Returns
        -------
        east, north, up : ~pygeoid.conventions.units.Quantity
            Local east-north-up cartesian coordinates.
        """
        east, north, up = transform.ecef_to_enu(self.x, self.y, self.z, origin, ell=ell)

        return east, north, up

    def represent_as(self, base, s="base", in_frame_units=False):
        if (
            inspect.isclass(base) and issubclass(base, GeodeticRepresentation)
        ) or base == "geodetic":
            return self.geodetic
        elif (
            inspect.isclass(base)
            and issubclass(base, EllipsoidalHarmonicRepresentation)
        ) or base == "ellipsoidalharmonic":
            return self.ellipsoidal_harmonic
        else:
            return super().represent_as(base, s=s, in_frame_units=in_frame_units)
