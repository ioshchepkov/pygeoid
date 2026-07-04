"""Lightweight containers for common geodetic coordinate systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

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


@dataclass(frozen=True, slots=True)
class EllipsoidalCoordinates(Coordinates, ABC):
    """Base class for coordinates referenced to an ellipsoid."""

    ellipsoid: Ellipsoid = field(default_factory=Ellipsoid, kw_only=True)


@dataclass(frozen=True, slots=True)
class CartesianCoordinates(Coordinates):
    """Earth-centred Cartesian coordinates."""

    x: u.Quantity
    y: u.Quantity
    z: u.Quantity

    @classmethod
    def from_cartesian(cls, coordinates: "CartesianCoordinates") -> Self:
        """Create a Cartesian coordinate container."""
        return cls(coordinates.x, coordinates.y, coordinates.z)

    def to_cartesian(self) -> "CartesianCoordinates":
        """Return these Cartesian coordinates."""
        return self


@dataclass(frozen=True, slots=True)
class SphericalCoordinates(Coordinates):
    """Geocentric spherical coordinates."""

    lat: u.Quantity
    lon: u.Quantity
    radius: u.Quantity

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


@dataclass(frozen=True, slots=True)
class GeodeticCoordinates(EllipsoidalCoordinates):
    """Geodetic coordinates referenced to an ellipsoid."""

    lat: u.Quantity
    lon: u.Quantity
    height: u.Quantity

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


@dataclass(frozen=True, slots=True)
class EllipsoidalHarmonicCoordinates(EllipsoidalCoordinates):
    """Ellipsoidal-harmonic coordinates referenced to an ellipsoid."""

    rlat: u.Quantity
    lon: u.Quantity
    u_ax: u.Quantity

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
