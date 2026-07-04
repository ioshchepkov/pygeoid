import astropy.units as u
import numpy as np

from pygeoid.geometry.coordinates import (
    CartesianCoordinates,
    Coordinates,
    EllipsoidalCoordinates,
    EllipsoidalHarmonicCoordinates,
    GeodeticCoordinates,
    SphericalCoordinates,
)
from pygeoid.geometry.ellipsoid import Ellipsoid


def assert_cartesian_equal(actual, expected):
    np.testing.assert_allclose(
        [actual.x.to_value(u.m), actual.y.to_value(u.m), actual.z.to_value(u.m)],
        [expected.x.to_value(u.m), expected.y.to_value(u.m), expected.z.to_value(u.m)],
        atol=1e-5,
    )


def test_coordinate_hierarchy():
    assert issubclass(CartesianCoordinates, Coordinates)
    assert issubclass(SphericalCoordinates, Coordinates)
    assert issubclass(GeodeticCoordinates, EllipsoidalCoordinates)
    assert issubclass(EllipsoidalHarmonicCoordinates, EllipsoidalCoordinates)


def test_cartesian_roundtrip():
    coordinates = CartesianCoordinates(1 * u.m, 2 * u.m, 3 * u.m)

    assert coordinates.to_cartesian() is coordinates
    assert CartesianCoordinates.from_cartesian(coordinates) == coordinates


def test_spherical_roundtrip():
    cartesian = CartesianCoordinates(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)

    assert_cartesian_equal(
        SphericalCoordinates.from_cartesian(cartesian).to_cartesian(), cartesian
    )


def test_geodetic_roundtrip_preserves_ellipsoid():
    cartesian = CartesianCoordinates(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)
    ellipsoid = Ellipsoid("intl")

    coordinates = GeodeticCoordinates.from_cartesian(cartesian, ellipsoid=ellipsoid)

    assert coordinates.ellipsoid is ellipsoid
    assert_cartesian_equal(coordinates.to_cartesian(), cartesian)


def test_ellipsoidal_harmonic_roundtrip_preserves_ellipsoid():
    cartesian = CartesianCoordinates(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)
    ellipsoid = Ellipsoid("intl")

    coordinates = EllipsoidalHarmonicCoordinates.from_cartesian(
        cartesian, ellipsoid=ellipsoid
    )

    assert coordinates.ellipsoid is ellipsoid
    assert_cartesian_equal(coordinates.to_cartesian(), cartesian)


def test_ellipsoidal_coordinates_get_independent_default_ellipsoids():
    first = GeodeticCoordinates(1 * u.deg, 2 * u.deg, 3 * u.m)
    second = GeodeticCoordinates(1 * u.deg, 2 * u.deg, 3 * u.m)

    assert first.ellipsoid is not second.ellipsoid
