"""Geocentric spherical coordinates."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from astropy.coordinates.angles import Latitude, Longitude

from pygeoid.conventions import units as u

from . import transform
from .base import BaseCoordinates
from .cartesian import CartesianCoordinates

__all__ = ["SphericalCoordinates"]


@dataclass(frozen=True, slots=True)
class SphericalCoordinates(BaseCoordinates):
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
        shape = np.broadcast_shapes(self.lat.shape, self.lon.shape, self.radius.shape)
        zero = np.zeros(shape, dtype=float) * u.one
        return {
            "lon": (-sinlon, coslon, zero),
            "lat": (-sinlat * coslon, -sinlat * sinlon, coslat),
            "radius": (coslat * coslon, coslat * sinlon, sinlat),
        }

    def scale_factors(self):
        """Return spherical scale factors."""
        sf_lat = self.radius / u.radian
        sf_lon = self.radius * np.cos(self.lat) / u.radian
        shape = np.broadcast_shapes(self.lat.shape, self.lon.shape, self.radius.shape)
        sf_radius = np.ones(shape, dtype=float) * u.one
        return {"lon": sf_lon, "lat": sf_lat, "radius": sf_radius}
