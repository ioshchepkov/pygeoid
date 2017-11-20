
import pytest
import numpy as np
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.transform import *


ell = Ellipsoid('GRS80')

# test data
n_test = 10  # * 2
r_ = np.geomspace(1, 1e8, num=n_test)
r_ = np.append(-r_[::-1], r_)
x, y, z = np.meshgrid(r_, r_, r_, indexing='ij')


def test_cartesian_to_geodetic_and_back():
    lat, lon, height = cartesian_to_geodetic(x, y, z, ell)
    b_x, b_y, b_z = geodetic_to_cartesian(lat, lon, height, ell)

    assert np.allclose(x, b_x)
    assert np.allclose(y, b_y)
    assert np.allclose(z, b_z)


def test_cartesian_to_spherical_and_back():
    lat, lon, radius = cartesian_to_spherical(x, y, z)
    b_x, b_y, b_z = spherical_to_cartesian(lat, lon, radius)

    assert np.allclose(x, b_x)
    assert np.allclose(y, b_y)
    assert np.allclose(z, b_z)


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

    rlat, lon, u = cartesian_to_ellipsoidal(x_, y_, z_, ell)
    b_x, b_y, b_z = ellipsoidal_to_cartesian(rlat, lon, u, ell)

    assert np.allclose(x_, b_x)
    assert np.allclose(y_, b_y)
    assert np.allclose(z_, b_z)


def test_geodetic_to_spherical_and_back():
    lat, lon, height = cartesian_to_geodetic(x, y, z, ell)

    sph_lat, sph_lon, radius = geodetic_to_spherical(lat, lon, height, ell)
    assert np.allclose(lon, sph_lon)
    b_lat, b_lon, b_height = spherical_to_geodetic(
        sph_lat, sph_lon, radius, ell)

    assert np.allclose(lat, b_lat)
    assert np.allclose(lon, b_lon)
    assert np.allclose(height, b_height)


def test_geodetic_to_elliposidal_and_back():
    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2
    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    lat, lon, height = cartesian_to_geodetic(x_, y_, z_, ell)

    rlat, ell_lon, u = geodetic_to_ellipsoidal(lat, lon, height, ell)
    assert np.allclose(lon, ell_lon)

    b_lat, b_lon, b_height = ellipsoidal_to_geodetic(rlat, ell_lon, u, ell)

    assert np.allclose(lat, b_lat)
    assert np.allclose(lon, b_lon)
    assert np.allclose(height, b_height)
