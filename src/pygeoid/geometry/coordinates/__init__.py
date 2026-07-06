"""Lightweight containers for common geodetic coordinate systems."""

from .base import BaseCoordinates
from .cartesian import CartesianCoordinates
from .ellipsoidal import (
    EllipsoidalCoordinates,
    EllipsoidalHarmonicCoordinates,
    GeodeticCoordinates,
)
from .spherical import SphericalCoordinates

__all__ = [
    "BaseCoordinates",
    "EllipsoidalCoordinates",
    "CartesianCoordinates",
    "SphericalCoordinates",
    "GeodeticCoordinates",
    "EllipsoidalHarmonicCoordinates",
]
