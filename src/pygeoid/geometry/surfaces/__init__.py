"""Reference surfaces used in physical geodesy."""

from .base import Surface
from .ellipsoid import DEFAULT_ELLIPSOID, Ellipsoid, get_ellps_map
from .sphere import Sphere

__all__ = [
    "DEFAULT_ELLIPSOID",
    "Ellipsoid",
    "Sphere",
    "Surface",
    "get_ellps_map",
]
