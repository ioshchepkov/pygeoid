import astropy.units as u
import numpy as np

from pygeoid.geometry import Position
from pygeoid.geometry.ellipsoid import DEFAULT_ELLIPSOID, Ellipsoid
from pygeoid.geometry.frame import LocalTangentPlane
from pygeoid.geometry.representation import (
    EllipsoidalHarmonicRepresentation,
    GeodeticRepresentation,
)
from pygeoid.geometry.transform import ecef_to_enu, enu_to_ecef

ell = Ellipsoid("GRS80")

# test data
n_test = 10  # * 2
r_ = np.geomspace(1, 1e8, num=n_test)
r_ = np.append(-r_[::-1], r_)
x, y, z = np.meshgrid(r_, r_, r_, indexing="ij") * u.m
p = Position(x, y, z)


def test_cartesian():
    np.testing.assert_equal(x.value, p.x.value)
    np.testing.assert_equal(y.value, p.y.value)
    np.testing.assert_equal(z.value, p.z.value)

    np.testing.assert_equal([x.value, y.value, z.value], p.cartesian.xyz.value)


def test_default_ellipsoid_is_instance_attribute():
    p = Position(1 * u.m, 2 * u.m, 3 * u.m)
    geod = GeodeticRepresentation(37 * u.deg, 55 * u.deg, 100 * u.m)
    ellharm = EllipsoidalHarmonicRepresentation(
        55 * u.deg, 37 * u.deg, 1e7 * u.m
    )

    assert not hasattr(Position, "_ellipsoid")
    assert not hasattr(GeodeticRepresentation, "_ellipsoid")
    assert not hasattr(EllipsoidalHarmonicRepresentation, "_ellipsoid")
    assert p.ellipsoid is not geod.ellipsoid
    assert geod.ellipsoid is not ellharm.ellipsoid
    np.testing.assert_equal(p.ellipsoid.a.value, Ellipsoid(DEFAULT_ELLIPSOID).a.value)
    np.testing.assert_equal(geod.ellipsoid.a.value, Ellipsoid(DEFAULT_ELLIPSOID).a.value)
    np.testing.assert_equal(
        ellharm.ellipsoid.a.value, Ellipsoid(DEFAULT_ELLIPSOID).a.value
    )


def test_from_to_geodetic():
    b_p = Position.from_geodetic(
        p.geodetic.lat, p.geodetic.lon, p.geodetic.height, ell=ell
    )
    np.testing.assert_array_almost_equal(
        b_p.cartesian.xyz.value, [x.value, y.value, z.value], decimal=5
    )


def test_from_to_spherical():
    b_p = Position.from_spherical(p.spherical.lat, p.spherical.lon, p.spherical.distance)
    np.testing.assert_array_almost_equal(
        b_p.cartesian.xyz.value, [x.value, y.value, z.value], decimal=5
    )


def test_from_to_ellipsoidal():
    ell = Ellipsoid("intl")

    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    p = Position(x_, y_, z_, ell=ell)
    ellipsoidal_harmonic = p.ellipsoidal_harmonic
    b_p = Position.from_ellipsoidal_harmonic(
        ellipsoidal_harmonic.rlat,
        ellipsoidal_harmonic.lon,
        ellipsoidal_harmonic.u_ax,
        ell=ell,
    )
    b_x, b_y, b_z = b_p.cartesian.xyz.value

    np.testing.assert_array_almost_equal(
        [b_x, b_y, b_z], [x_.value, y_.value, z_.value], decimal=5
    )
    assert b_p.ellipsoid is ell


def test_from_to_ellipsoidal_default_ellipsoid():
    p = Position(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)

    b_p = Position.from_ellipsoidal_harmonic(
        p.ellipsoidal_harmonic.rlat,
        p.ellipsoidal_harmonic.lon,
        p.ellipsoidal_harmonic.u_ax,
    )

    np.testing.assert_allclose(b_p.cartesian.xyz.value, p.cartesian.xyz.value)


def test_represent_as_preserves_frame_ellipsoid():
    ell1 = Ellipsoid("GRS80")
    ell2 = Ellipsoid("intl")

    p1 = Position(1e7 * u.m, 2e7 * u.m, 3e7 * u.m, ell=ell1)
    p2 = Position(1e7 * u.m, 2e7 * u.m, 3e7 * u.m, ell=ell2)

    assert p1.represent_as("geodetic").ellipsoid is ell1
    assert p2.represent_as("geodetic").ellipsoid is ell2
    assert p1.represent_as("ellipsoidalharmonic").ellipsoid is ell1
    assert p2.represent_as("ellipsoidalharmonic").ellipsoid is ell2


def test_to_enu_ell():
    lat0, lon0, height0 = 55.0 * u.deg, 37.0 * u.deg, 100.0 * u.m
    origin = (lat0, lon0, height0)
    enu = p.enu(origin=origin, ell=ell)

    b_x, b_y, b_z = enu_to_ecef(*enu, origin=origin, ell=ell)

    x_, y_, z_ = p.cartesian.xyz

    np.testing.assert_array_almost_equal(b_x.value, x_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z_.value, decimal=5)


def test_enu_uses_given_ellipsoid():
    ell = Ellipsoid("intl")
    point = Position(1e7 * u.m, 2e7 * u.m, 3e7 * u.m)
    origin = (55.0 * u.deg, 37.0 * u.deg, 100.0 * u.m)

    expected = ecef_to_enu(point.x, point.y, point.z, origin=origin, ell=ell)
    actual = point.enu(origin=origin, ell=ell)

    np.testing.assert_allclose(actual[0].value, expected[0].value)
    np.testing.assert_allclose(actual[1].value, expected[1].value)
    np.testing.assert_allclose(actual[2].value, expected[2].value)


def test_local_tangent_plane_transform_roundtrip():
    origin = Position.from_geodetic(55.0 * u.deg, 37.0 * u.deg, 100.0 * u.m)
    point = Position.from_geodetic(55.001 * u.deg, 37.002 * u.deg, 120.0 * u.m)
    local = LocalTangentPlane(origin=origin)

    local_point = point.transform_to(local)
    back = local_point.transform_to(Position())

    np.testing.assert_allclose(
        back.cartesian.xyz.to_value(u.m), point.cartesian.xyz.to_value(u.m)
    )


def test_local_tangent_plane_to_local_transform_roundtrip():
    origin0 = Position.from_geodetic(55.0 * u.deg, 37.0 * u.deg, 100.0 * u.m)
    origin1 = Position.from_geodetic(55.01 * u.deg, 37.01 * u.deg, 90.0 * u.m)
    point = Position.from_geodetic(55.001 * u.deg, 37.002 * u.deg, 120.0 * u.m)
    local0 = LocalTangentPlane(origin=origin0)
    local1 = LocalTangentPlane(origin=origin1)

    local1_point = point.transform_to(local0).transform_to(local1)
    back = local1_point.transform_to(Position())

    np.testing.assert_allclose(
        back.cartesian.xyz.to_value(u.m), point.cartesian.xyz.to_value(u.m)
    )
