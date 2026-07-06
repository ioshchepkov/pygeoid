"""Coordinates referenced to an ellipsoid."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from astropy.coordinates.angles import Latitude, Longitude

from pygeoid.conventions import units as u
from pygeoid.geometry.surfaces import Ellipsoid

from . import transform
from .base import BaseCoordinates
from .cartesian import CartesianCoordinates

__all__ = [
    "EllipsoidalCoordinates",
    "GeodeticCoordinates",
    "EllipsoidalHarmonicCoordinates",
]


@dataclass(frozen=True, slots=True)
class EllipsoidalCoordinates(BaseCoordinates, ABC):
    """Base class for coordinates referenced to an ellipsoid."""

    ellipsoid: Ellipsoid = field(default_factory=Ellipsoid, kw_only=True)


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
        shape = np.broadcast_shapes(self.lat.shape, self.lon.shape, self.height.shape)
        zero = np.zeros(shape, dtype=float) * u.one
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
        shape = np.broadcast_shapes(self.lat.shape, self.lon.shape, self.height.shape)
        sf_height = np.ones(shape, dtype=float) * u.one
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

        shape = np.broadcast_shapes(self.rlat.shape, self.lon.shape, self.u_ax.shape)
        zero = np.zeros(shape, dtype=float) * u.one

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
