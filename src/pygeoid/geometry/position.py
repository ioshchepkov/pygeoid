"""Position container with lazy coordinate transformations."""

from functools import cached_property

import numpy as np

from pygeoid.conventions import units as u
from pygeoid.geometry.coordinates import (
    BaseCoordinates,
    CartesianCoordinates,
    EllipsoidalCoordinates,
    EllipsoidalHarmonicCoordinates,
    GeodeticCoordinates,
    SphericalCoordinates,
    transform,
)
from pygeoid.geometry.ellipsoid import Ellipsoid

__all__ = ["Position"]


class Position:
    """A position represented by one initial set of coordinates.

    Coordinate transformations are performed only when another representation
    is requested, then cached on the instance.

    Parameters
    ----------
    coordinates
        Initial coordinate container.
    ell : Ellipsoid, optional
        Reference ellipsoid used to derive ellipsoidal coordinates from
        Cartesian or spherical coordinates.
    """

    def __init__(self, coordinates, *, ell=None):
        if not isinstance(coordinates, BaseCoordinates):
            raise TypeError("Position requires a BaseCoordinates instance.")

        if isinstance(coordinates, EllipsoidalCoordinates):
            if ell is not None and ell is not coordinates.ellipsoid:
                raise ValueError(
                    "ell conflicts with the ellipsoid stored in the coordinates."
                )
            ellipsoid = coordinates.ellipsoid
        else:
            ellipsoid = Ellipsoid() if ell is None else ell

        if not isinstance(ellipsoid, Ellipsoid):
            raise TypeError("ell must be an Ellipsoid instance.")

        self._coordinates = coordinates
        self._ellipsoid = ellipsoid

    @staticmethod
    def _angle(value):
        return value if isinstance(value, u.Quantity) else u.Quantity(value, u.deg)

    @staticmethod
    def _length(value):
        return value if isinstance(value, u.Quantity) else u.Quantity(value, u.m)

    @property
    def coordinates(self):
        """The initial, untransformed coordinate container."""
        return self._coordinates

    @property
    def ellipsoid(self):
        """Reference ellipsoid used for ellipsoidal transformations."""
        return self._ellipsoid

    @cached_property
    def cartesian(self):
        """Cartesian coordinates, transformed lazily."""
        if isinstance(self.coordinates, CartesianCoordinates):
            return self.coordinates
        return self.coordinates.to_cartesian()

    @cached_property
    def spherical(self):
        """Geocentric spherical coordinates, transformed lazily."""
        if isinstance(self.coordinates, SphericalCoordinates):
            return self.coordinates
        return SphericalCoordinates.from_cartesian(self.cartesian)

    @cached_property
    def geodetic(self):
        """Geodetic coordinates, transformed lazily."""
        if isinstance(self.coordinates, GeodeticCoordinates):
            return self.coordinates
        return GeodeticCoordinates.from_cartesian(
            self.cartesian, ellipsoid=self.ellipsoid
        )

    @cached_property
    def ellipsoidal_harmonic(self):
        """Ellipsoidal-harmonic coordinates, transformed lazily."""
        if isinstance(self.coordinates, EllipsoidalHarmonicCoordinates):
            return self.coordinates
        return EllipsoidalHarmonicCoordinates.from_cartesian(
            self.cartesian, ellipsoid=self.ellipsoid
        )

    @property
    def x(self):
        return self.cartesian.x

    @property
    def y(self):
        return self.cartesian.y

    @property
    def z(self):
        return self.cartesian.z

    @property
    def shape(self):
        """Broadcast shape of the Cartesian coordinate components."""
        return np.broadcast_shapes(np.shape(self.x), np.shape(self.y), np.shape(self.z))

    @classmethod
    def from_spherical(cls, lat, lon, radius, *, ell=None):
        """Create a position retaining spherical coordinates."""
        coordinates = SphericalCoordinates(
            cls._angle(lat), cls._angle(lon), cls._length(radius)
        )
        return cls(coordinates, ell=ell)

    @classmethod
    def from_geodetic(cls, lat, lon, height=0.0, ell=None):
        """Create a position retaining geodetic coordinates."""
        ellipsoid = Ellipsoid() if ell is None else ell
        coordinates = GeodeticCoordinates(
            cls._angle(lat),
            cls._angle(lon),
            cls._length(height),
            ellipsoid=ellipsoid,
        )
        return cls(coordinates)

    @classmethod
    def from_ellipsoidal_harmonic(cls, rlat, lon, u_ax, ell=None):
        """Create a position retaining ellipsoidal-harmonic coordinates."""
        ellipsoid = Ellipsoid() if ell is None else ell
        coordinates = EllipsoidalHarmonicCoordinates(
            cls._angle(rlat),
            cls._angle(lon),
            cls._length(u_ax),
            ellipsoid=ellipsoid,
        )
        return cls(coordinates)

    @u.quantity_input
    def enu(self, origin: tuple[u.deg, u.deg, u.m], ell=None):
        """Return local east-north-up Cartesian coordinates."""
        return transform.ecef_to_enu(self.x, self.y, self.z, origin, ell=ell)

    def transform_to(self, frame):
        """Transform this position to a local tangent plane."""
        from pygeoid.geometry.frame import LocalTangentPlane

        if isinstance(frame, LocalTangentPlane):
            return frame.from_position(self)
        raise TypeError(f"Unsupported target frame: {frame!r}")

    def represent_as(self, coordinates):
        """Return coordinates in a supported coordinate system.

        ``coordinates`` may be a coordinate class or one of ``"cartesian"``,
        ``"spherical"``, ``"geodetic"`` or ``"ellipsoidalharmonic"``.
        """
        coordinate_types = {
            CartesianCoordinates: "cartesian",
            SphericalCoordinates: "spherical",
            GeodeticCoordinates: "geodetic",
            EllipsoidalHarmonicCoordinates: "ellipsoidal_harmonic",
        }
        if coordinates in coordinate_types:
            return getattr(self, coordinate_types[coordinates])

        if isinstance(coordinates, str):
            name = coordinates.lower().replace("_", "").replace("-", "")
            by_name = {
                "cartesian": "cartesian",
                "spherical": "spherical",
                "geodetic": "geodetic",
                "ellipsoidalharmonic": "ellipsoidal_harmonic",
            }
            try:
                return getattr(self, by_name[name])
            except KeyError:
                pass

        raise ValueError(f"Unsupported coordinate system: {coordinates!r}")
