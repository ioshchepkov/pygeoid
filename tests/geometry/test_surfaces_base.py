import pygeoid.geometry.surfaces.base as base
from pygeoid.geometry.surfaces import Ellipsoid, Sphere, Surface


def test_surface_hierarchy():
    assert base.__all__ == ["Surface"]
    assert issubclass(Ellipsoid, Surface)
    assert issubclass(Sphere, Surface)
