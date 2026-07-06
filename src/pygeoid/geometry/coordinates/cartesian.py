"""Earth-centred Cartesian coordinates."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from pygeoid.conventions import units as u

from .base import BaseCoordinates

__all__ = ["CartesianCoordinates"]


@dataclass(frozen=True, slots=True)
class CartesianCoordinates(BaseCoordinates):
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
        shape = np.broadcast_shapes(self.x.shape, self.y.shape, self.z.shape)
        zero = np.zeros(shape, dtype=float) * u.one
        one = np.ones(shape, dtype=float) * u.one
        return {
            "x": (one, zero, zero),
            "y": (zero, one, zero),
            "z": (zero, zero, one),
        }

    def scale_factors(self):
        """Return the Cartesian scale factors."""
        shape = np.broadcast_shapes(self.x.shape, self.y.shape, self.z.shape)
        one = np.ones(shape, dtype=float) * u.one
        return {"x": one, "y": one, "z": one}
