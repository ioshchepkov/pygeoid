"""Lightweight containers for common geodetic coordinate systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from astropy.coordinates.angles import Latitude, Longitude

from pygeoid.conventions import units as u
from pygeoid.geometry import transform
from pygeoid.geometry.ellipsoid import Ellipsoid

__all__ = [
    "Coordinates",
    "EllipsoidalCoordinates",
    "CartesianCoordinates",
    "SphericalCoordinates",
    "GeodeticCoordinates",
    "EllipsoidalHarmonicCoordinates",
]


def _broadcast_shape(*quantities):
    shape = ()
    for quantity in quantities:
        shape = np.broadcast_shapes(shape, quantity.shape)
    return shape


def _zeros(shape):
    return np.zeros(shape, dtype=float) * u.one


def _ones(shape):
    return np.ones(shape, dtype=float) * u.one


@dataclass(frozen=True, slots=True)
class Coordinates(ABC):
    """Base class for coordinate containers."""

    @classmethod
    @abstractmethod
    def from_cartesian(cls, coordinates: "CartesianCoordinates") -> Self:
        """Create coordinates from Cartesian coordinates."""

    @abstractmethod
    def to_cartesian(self) -> "CartesianCoordinates":
        """Convert these coordinates to Cartesian coordinates."""

    @abstractmethod
    def unit_vectors(self):
        """Return the local basis vectors as Cartesian components."""

    @abstractmethod
    def scale_factors(self):
        """Return the metric scale factors for the coordinate system."""


@dataclass(frozen=True, slots=True)
class EllipsoidalCoordinates(Coordinates, ABC):
    """Base class for coordinates referenced to an ellipsoid."""

    ellipsoid: Ellipsoid = field(default_factory=Ellipsoid, kw_only=True)


@dataclass(frozen=True, slots=True)
class CartesianCoordinates(Coordinates):
    """Earth-centred Cartesian coordinates."""

    x: u.Quantity[u.m]
    y: u.Quantity[u.m]
    z: u.Quantity[u.m]

    @classmethod
    def from_cartesian(cls, coordinates: "CartesianCoordinates") -> Self:
        """Create a Cartesian coordinate container."""
        return cls(coordinates.x, coordinates.y, coordinates.z)

    def to_cartesian(self) -> "CartesianCoordinates":
        """Return these Cartesian coordinates."""
        return self

    def unit_vectors(self):
        """Return the orthonormal Cartesian basis."""
        shape = _broadcast_shape(self.x, self.y, self.z)
        zero = _zeros(shape)
        one = _ones(shape)
        return {
            "x": (one, zero, zero),
            "y": (zero, one, zero),
            "z": (zero, zero, one),
        }

    def scale_factors(self):
        """Return the Cartesian scale factors."""
        shape = _broadcast_shape(self.x, self.y, self.z)
        one = _ones(shape)
        return {"x": one, "y": one, "z": one}


@dataclass(frozen=True, slots=True)
class SphericalCoordinates(Coordinates):
    """Geocentric spherical coordinates."""

    lat: Latitude
    lon: Longitude
    radius: u.Quantity[u.m]

    @classmethod
    def from_cartesian(cls, coordinates: CartesianCoordinates) -> Self:
        """Convert Cartesian coordinates to spherical coordinates."""
        lat, lon, radius = transform.cartesian_to_spherical(
            coordinates.x, coordinates.y, coordinates.z
        )
        return cls(lat, lon, radius)

    def to_cartesian(self) -> CartesianCoordinates:
        """Convert spherical coordinates to Cartesian coordinates."""
        return CartesianCoordinates(
            *transform.spherical_to_cartesian(self.lat, self.lon, self.radius)
        )

    def unit_vectors(self):
        """Return spherical basis vectors in Cartesian components."""
        sinlon, coslon = np.sin(self.lon), np.cos(self.lon)
        sinlat, coslat = np.sin(self.lat), np.cos(self.lat)
        shape = _broadcast_shape(self.lat, self.lon, self.radius)
        zero = _zeros(shape)
        return {
            "lon": (-sinlon, coslon, zero),
            "lat": (-sinlat * coslon, -sinlat * sinlon, coslat),
            "radius": (coslat * coslon, coslat * sinlon, sinlat),
        }

    def scale_factors(self):
        """Return spherical scale factors."""
        sf_lat = self.radius / u.radian
        sf_lon = self.radius * np.cos(self.lat) / u.radian
        shape = _broadcast_shape(self.lat, self.lon, self.radius)
        sf_radius = _ones(shape)
        return {"lon": sf_lon, "lat": sf_lat, "radius": sf_radius}


@dataclass(frozen=True, slots=True)
class GeodeticCoordinates(EllipsoidalCoordinates):
    """Geodetic coordinates referenced to an ellipsoid."""

    lat: Latitude
    lon: Longitude
    height: u.Quantity[u.m]

    @classmethod
    def from_cartesian(
        cls,
        coordinates: CartesianCoordinates,
        *,
        ellipsoid: Ellipsoid | None = None,
    ) -> Self:
        """Convert Cartesian coordinates to geodetic coordinates."""
        if ellipsoid is None:
            ellipsoid = Ellipsoid()
        lat, lon, height = transform.cartesian_to_geodetic(
            coordinates.x, coordinates.y, coordinates.z, ellipsoid
        )
        return cls(lat, lon, height, ellipsoid=ellipsoid)

    def to_cartesian(self) -> CartesianCoordinates:
        """Convert geodetic coordinates to Cartesian coordinates."""
        return CartesianCoordinates(
            *transform.geodetic_to_cartesian(
                self.lat, self.lon, self.height, self.ellipsoid
            )
        )

    def unit_vectors(self):
        """Return geodetic basis vectors in Cartesian components."""
        sinlon, coslon = np.sin(self.lon), np.cos(self.lon)
        sinlat, coslat = np.sin(self.lat), np.cos(self.lat)
        shape = _broadcast_shape(self.lat, self.lon, self.height)
        zero = _zeros(shape)
        return {
            "lon": (-sinlon, coslon, zero),
            "lat": (-sinlat * coslon, -sinlat * sinlon, coslat),
            "height": (coslat * coslon, coslat * sinlon, sinlat),
        }

    def scale_factors(self):
        """Return geodetic scale factors."""
        pmer_rad = self.ellipsoid.meridian_curvature_radius(self.lat)
        sf_lat = (pmer_rad + self.height) / u.radian
        pver_rad = self.ellipsoid.prime_vertical_curvature_radius(self.lat)
        sf_lon = (pver_rad + self.height) * np.cos(self.lat) / u.radian
        shape = _broadcast_shape(self.lat, self.lon, self.height)
        sf_height = _ones(shape)
        return {"lon": sf_lon, "lat": sf_lat, "height": sf_height}


@dataclass(frozen=True, slots=True)
class EllipsoidalHarmonicCoordinates(EllipsoidalCoordinates):
    """Ellipsoidal-harmonic coordinates referenced to an ellipsoid."""

    rlat: Latitude
    lon: Longitude
    u_ax: u.Quantity[u.m]

    @classmethod
    def from_cartesian(
        cls,
        coordinates: CartesianCoordinates,
        *,
        ellipsoid: Ellipsoid | None = None,
    ) -> Self:
        """Convert Cartesian coordinates to ellipsoidal-harmonic coordinates."""
        if ellipsoid is None:
            ellipsoid = Ellipsoid()
        rlat, lon, u_ax = transform.cartesian_to_ellipsoidal(
            coordinates.x, coordinates.y, coordinates.z, ellipsoid
        )
        return cls(rlat, lon, u_ax, ellipsoid=ellipsoid)

    def to_cartesian(self) -> CartesianCoordinates:
        """Convert ellipsoidal-harmonic coordinates to Cartesian coordinates."""
        return CartesianCoordinates(
            *transform.ellipsoidal_to_cartesian(
                self.rlat, self.lon, self.u_ax, self.ellipsoid
            )
        )

    def unit_vectors(self):
        """Return ellipsoidal-harmonic basis vectors in Cartesian components."""
        sinlon, coslon = np.sin(self.lon), np.cos(self.lon)
        sinrlat, cosrlat = np.sin(self.rlat), np.cos(self.rlat)
        le2 = self.ellipsoid.linear_eccentricity**2
        u_ax2 = self.u_ax**2
        k = np.sqrt(u_ax2 + le2)

        w = np.sqrt(u_ax2 + le2 * sinrlat) / k
        uwk = self.u_ax / (w * k)

        shape = _broadcast_shape(self.rlat, self.lon, self.u_ax)
        zero = _zeros(shape)

        return {
            "lon": (-sinlon, coslon, zero),
            "rlat": (-sinrlat * coslon / w, -sinrlat * sinlon / w, uwk * cosrlat),
            "u_ax": (
                uwk * cosrlat * coslon,
                uwk * cosrlat * sinlon,
                sinrlat / w,
            ),
        }

    def scale_factors(self):
        """Return ellipsoidal-harmonic scale factors."""
        le2 = self.ellipsoid.linear_eccentricity**2
        u_ax2 = self.u_ax**2
        k = np.sqrt(u_ax2 + le2)

        sf_rlat = np.sqrt(u_ax2 + le2 * np.sin(self.rlat) ** 2) / u.radian
        sf_u_ax = sf_rlat / k
        sf_lon = k * np.cos(self.rlat) / u.radian

        return {"lon": sf_lon, "rlat": sf_rlat, "u_ax": sf_u_ax}
