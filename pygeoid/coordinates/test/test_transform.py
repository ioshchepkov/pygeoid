
import pytest
import numpy as np
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.transform import *


ell = Ellipsoid('GRS80')

# test data
n_test = 10  # * 2
r_ = np.geomspace(1, 1e8, num=n_test)
r_ = np.append(-r_[::-1], r_)
x, y, z = np.meshgrid(r_, r_, r_, indexing='ij') * u.m


def test_cartesian_to_geodetic_and_back():
    lat, lon, height = cartesian_to_geodetic(x, y, z, ell)
    b_x, b_y, b_z = geodetic_to_cartesian(lat, lon, height, ell)

    np.testing.assert_array_almost_equal(b_x.value, x.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z.value, decimal=5)


def test_cartesian_to_spherical_and_back():
    lat, lon, radius = cartesian_to_spherical(x, y, z)
    b_x, b_y, b_z = spherical_to_cartesian(lat, lon, radius)

    np.testing.assert_array_almost_equal(b_x.value, x.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z.value, decimal=5)


def test_focal_distance_exception():
    cond = (x**2 + y**2 + z**2) >= ell.linear_eccentricity**2

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    with pytest.raises(ValueError):
        rlat, lon, u = cartesian_to_ellipsoidal(x_, y_, z_, ell)


def test_cartesian_to_ellipsoidal_and_back():
    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    rlat, lon, u_ax = cartesian_to_ellipsoidal(x_, y_, z_, ell)
    b_x, b_y, b_z = ellipsoidal_to_cartesian(rlat, lon, u_ax, ell)

    np.testing.assert_array_almost_equal(b_x.value, x_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z_.value, decimal=5)


def test_geodetic_to_spherical_and_back():
    lat, lon, height = cartesian_to_geodetic(x, y, z, ell)

    sph_lat, sph_lon, radius = geodetic_to_spherical(lat, lon, height, ell)
    np.testing.assert_array_almost_equal(
            sph_lon.to('degree').value,
            lon.to('degree').value, decimal=9)

    b_lat, b_lon, b_height = spherical_to_geodetic(
        sph_lat, sph_lon, radius, ell)

    np.testing.assert_array_almost_equal(b_lat.value, lat.value, decimal=9)
    np.testing.assert_array_almost_equal(b_lon.value, lon.value, decimal=9)
    np.testing.assert_array_almost_equal(b_height.value, height.value, decimal=5)


def test_geodetic_to_elliposidal_and_back():
    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2
    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    lat, lon, height = cartesian_to_geodetic(x_, y_, z_, ell)

    rlat, ell_lon, u_ax = geodetic_to_ellipsoidal(lat, lon, height, ell)
    np.testing.assert_array_almost_equal(
            ell_lon.to('degree').value,
            lon.to('degree').value, decimal=9)

    b_lat, b_lon, b_height = ellipsoidal_to_geodetic(rlat, ell_lon, u_ax, ell)

    np.testing.assert_array_almost_equal(b_lat.to('degree').value, lat.to('degree').value, decimal=9)
    np.testing.assert_array_almost_equal(b_lon.to('degree').value, lon.to('degree').value, decimal=9)
    np.testing.assert_array_almost_equal(b_height.value, height.value, decimal=5)


def test_ecef_to_enef_and_back_ell():
    x_, y_, z_ = x, y, z

    lat0, lon0, height0 = 55.0*u.deg, 37.0*u.deg, 100.0 *u.m

    enu = ecef_to_enu(x_, y_, z_, origin=(lat0, lon0, height0), ell=ell)

    b_x, b_y, b_z = enu_to_ecef(*enu, origin=(lat0, lon0, height0), ell=ell)

    np.testing.assert_array_almost_equal(b_x.value, x_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z_.value, decimal=5)

def test_ecef_to_enef_and_back_sph():
    x_, y_, z_ = x, y, z

    lat0, lon0, height0 = 55.0*u.deg, 37.0*u.deg, 100.0 *u.m

    enu = ecef_to_enu(x_, y_, z_, origin=(lat0, lon0, height0))

    b_x, b_y, b_z = enu_to_ecef(*enu, origin=(lat0, lon0, height0))

    np.testing.assert_array_almost_equal(b_x.value, x_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z_.value, decimal=5)

def test_geodetic_to_enu_and_back():
    lat, lon, height = cartesian_to_geodetic(x, y, z, ell)

    origin = (55.0 * u.deg, 37.0 * u.deg, 100.0 * u.m)
    enu = geodetic_to_enu(lat, lon, height, origin, ell=ell)
    b_lat, b_lon, b_height = enu_to_geodetic(
        *enu, origin, ell)

    # NOTE: Why decimal=6?
    np.testing.assert_array_almost_equal(b_lat.to('degree').value,
            lat.to('degree').value, decimal=6)
    np.testing.assert_array_almost_equal(b_lon.to('degree').value,
            lon.to('degree').value, decimal=6)
    np.testing.assert_array_almost_equal(b_height.value, height.value, decimal=5)

def test_cartesian_to_polar_and_back():
    values = np.round(np.linspace(-100000, 100000, 1000), 5)
    x, y = np.meshgrid(values, values, indexing='ij') * u.m
    theta, radius = cartesian_to_polar(x, y)
    b_x, b_y = polar_to_cartesian(theta, radius)

    np.testing.assert_array_almost_equal(b_x.value, x.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y.value, decimal=5)
