import astropy.units as u
import numpy as np

from pygeoid.geometry.coordinates import (
    BaseCoordinates,
    CartesianCoordinates,
    EllipsoidalCoordinates,
    EllipsoidalHarmonicCoordinates,
    GeodeticCoordinates,
    SphericalCoordinates,
)
from pygeoid.geometry.surfaces import Ellipsoid


def assert_cartesian_equal(actual, expected):
    np.testing.assert_allclose(
        [actual.x.to_value(u.m), actual.y.to_value(u.m), actual.z.to_value(u.m)],
        [expected.x.to_value(u.m), expected.y.to_value(u.m), expected.z.to_value(u.m)],
        atol=1e-5,
    )


def assert_vector_equal(actual, expected):
    for key, expected_component in expected.items():
        np.testing.assert_allclose(
            actual[key][0].to_value(u.one),
            expected_component[0].to_value(u.one),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual[key][1].to_value(u.one),
            expected_component[1].to_value(u.one),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual[key][2].to_value(u.one),
            expected_component[2].to_value(u.one),
            atol=1e-12,
        )


def assert_scale_factors_equal(actual, expected):
    for key, expected_value in expected.items():
        np.testing.assert_allclose(
            actual[key].to_value(expected_value.unit),
            expected_value.to_value(expected_value.unit),
            atol=1e-12,
        )


def test_coordinate_hierarchy():
    assert issubclass(CartesianCoordinates, BaseCoordinates)
    assert issubclass(SphericalCoordinates, BaseCoordinates)
    assert issubclass(GeodeticCoordinates, EllipsoidalCoordinates)
    assert issubclass(EllipsoidalHarmonicCoordinates, EllipsoidalCoordinates)


def test_cartesian_roundtrip():
    coordinates = CartesianCoordinates(1 * u.m, 2 * u.m, 3 * u.m)

    assert coordinates.to_cartesian() is coordinates
    assert CartesianCoordinates.from_cartesian(coordinates) == coordinates


def test_cartesian_unit_vectors_and_scale_factors():
    coordinates = CartesianCoordinates(1 * u.m, 2 * u.m, 3 * u.m)

    assert_vector_equal(
        coordinates.unit_vectors(),
        {
            "x": (1 * u.one, 0 * u.one, 0 * u.one),
            "y": (0 * u.one, 1 * u.one, 0 * u.one),
            "z": (0 * u.one, 0 * u.one, 1 * u.one),
        },
    )
    assert_scale_factors_equal(
        coordinates.scale_factors(),
        {"x": 1 * u.one, "y": 1 * u.one, "z": 1 * u.one},
    )


def test_spherical_roundtrip():
    cartesian = CartesianCoordinates(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)

    assert_cartesian_equal(
        SphericalCoordinates.from_cartesian(cartesian).to_cartesian(), cartesian
    )


def test_spherical_unit_vectors_and_scale_factors():
    coordinates = SphericalCoordinates(30 * u.deg, 40 * u.deg, 2 * u.m)

    sinlon, coslon = np.sin(coordinates.lon), np.cos(coordinates.lon)
    sinlat, coslat = np.sin(coordinates.lat), np.cos(coordinates.lat)

    assert_vector_equal(
        coordinates.unit_vectors(),
        {
            "lon": (-sinlon, coslon, 0 * u.one),
            "lat": (-sinlat * coslon, -sinlat * sinlon, coslat),
            "radius": (coslat * coslon, coslat * sinlon, sinlat),
        },
    )
    assert_scale_factors_equal(
        coordinates.scale_factors(),
        {
            "lon": coordinates.radius * np.cos(coordinates.lat) / u.radian,
            "lat": coordinates.radius / u.radian,
            "radius": 1 * u.one,
        },
    )


def test_geodetic_roundtrip_preserves_ellipsoid():
    cartesian = CartesianCoordinates(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)
    ellipsoid = Ellipsoid("intl")

    coordinates = GeodeticCoordinates.from_cartesian(cartesian, ellipsoid=ellipsoid)

    assert coordinates.ellipsoid is ellipsoid
    assert_cartesian_equal(coordinates.to_cartesian(), cartesian)


def test_geodetic_unit_vectors_and_scale_factors():
    ellipsoid = Ellipsoid("intl")
    coordinates = GeodeticCoordinates(30 * u.deg, 40 * u.deg, 100 * u.m, ellipsoid=ellipsoid)

    sinlon, coslon = np.sin(coordinates.lon), np.cos(coordinates.lon)
    sinlat, coslat = np.sin(coordinates.lat), np.cos(coordinates.lat)

    assert_vector_equal(
        coordinates.unit_vectors(),
        {
            "lon": (-sinlon, coslon, 0 * u.one),
            "lat": (-sinlat * coslon, -sinlat * sinlon, coslat),
            "height": (coslat * coslon, coslat * sinlon, sinlat),
        },
    )
    assert_scale_factors_equal(
        coordinates.scale_factors(),
        {
            "lon": (
                ellipsoid.prime_vertical_curvature_radius(coordinates.lat)
                + coordinates.height
            )
            * np.cos(coordinates.lat)
            / u.radian,
            "lat": (
                ellipsoid.meridian_curvature_radius(coordinates.lat)
                + coordinates.height
            )
            / u.radian,
            "height": 1 * u.one,
        },
    )


def test_ellipsoidal_harmonic_roundtrip_preserves_ellipsoid():
    cartesian = CartesianCoordinates(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)
    ellipsoid = Ellipsoid("intl")

    coordinates = EllipsoidalHarmonicCoordinates.from_cartesian(
        cartesian, ellipsoid=ellipsoid
    )

    assert coordinates.ellipsoid is ellipsoid
    assert_cartesian_equal(coordinates.to_cartesian(), cartesian)


def test_ellipsoidal_harmonic_unit_vectors_and_scale_factors():
    ellipsoid = Ellipsoid("intl")
    coordinates = EllipsoidalHarmonicCoordinates(
        30 * u.deg, 40 * u.deg, 1e7 * u.m, ellipsoid=ellipsoid
    )

    sinlon, coslon = np.sin(coordinates.lon), np.cos(coordinates.lon)
    sinrlat, cosrlat = np.sin(coordinates.rlat), np.cos(coordinates.rlat)
    le2 = ellipsoid.linear_eccentricity**2
    u_ax2 = coordinates.u_ax**2
    k = np.sqrt(u_ax2 + le2)
    w = np.sqrt(u_ax2 + le2 * sinrlat) / k
    uwk = coordinates.u_ax / (w * k)

    assert_vector_equal(
        coordinates.unit_vectors(),
        {
            "lon": (-sinlon, coslon, 0 * u.one),
            "rlat": (-sinrlat * coslon / w, -sinrlat * sinlon / w, uwk * cosrlat),
            "u_ax": (
                uwk * cosrlat * coslon,
                uwk * cosrlat * sinlon,
                sinrlat / w,
            ),
        },
    )
    assert_scale_factors_equal(
        coordinates.scale_factors(),
        {
            "lon": k * np.cos(coordinates.rlat) / u.radian,
            "rlat": np.sqrt(u_ax2 + le2 * np.sin(coordinates.rlat) ** 2)
            / u.radian,
            "u_ax": (
                np.sqrt(u_ax2 + le2 * np.sin(coordinates.rlat) ** 2)
                / u.radian
                / k
            ),
        },
    )


def test_ellipsoidal_coordinates_get_independent_default_ellipsoids():
    first = GeodeticCoordinates(1 * u.deg, 2 * u.deg, 3 * u.m)
    second = GeodeticCoordinates(1 * u.deg, 2 * u.deg, 3 * u.m)

    assert first.ellipsoid is not second.ellipsoid
