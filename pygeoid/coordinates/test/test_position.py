
import pytest
import numpy as np
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.position import Position3D


ell = Ellipsoid('GRS80')

# test data
n_test = 10  # * 2
r_ = np.geomspace(1, 1e8, num=n_test)
r_ = np.append(-r_[::-1], r_)
x, y, z = np.meshgrid(r_, r_, r_, indexing='ij')
p = Position3D(x, y, z)


def test_cartesian():
    np.testing.assert_equal(x, p.x)
    np.testing.assert_equal(y, p.y)
    np.testing.assert_equal(z, p.z)

    np.testing.assert_equal([x, y, z], p.cartesian)


def test_from_to_geodetic():
    lat, lon, height = p.geodetic(ell)
    b_p = Position3D.from_geodetic(lat, lon, height, ell=ell)
    np.testing.assert_array_almost_equal(b_p.cartesian,
                                         [x, y, z], decimal=5)


def test_from_to_spherical():
    lat, lon, radius = p.spherical()
    b_p = Position3D.from_spherical(lat, lon, radius)
    np.testing.assert_array_almost_equal(b_p.cartesian,
                                         [x, y, z], decimal=5)


def test_from_to_ellipsoidal():
    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    p = Position3D(x_, y_, z_)
    rlat, lon, u = p.ellipsoidal(ell=ell)
    b_p = Position3D.from_ellipsoidal(rlat, lon, u, ell=ell)
    np.testing.assert_array_almost_equal(b_p.cartesian,
                                         [x_, y_, z_], decimal=5)
