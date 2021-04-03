
import pytest
import numpy as np
import astropy.units as u
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.transform import enu_to_ecef
from pygeoid.coordinates.position import Position3D


ell = Ellipsoid('GRS80')

# test data
n_test = 10  # * 2
r_ = np.geomspace(1, 1e8, num=n_test)
r_ = np.append(-r_[::-1], r_)
x, y, z = np.meshgrid(r_, r_, r_, indexing='ij') * u.m
p = Position3D(x, y, z)


def test_cartesian():
    np.testing.assert_equal(x.value, p.x.value)
    np.testing.assert_equal(y.value, p.y.value)
    np.testing.assert_equal(z.value, p.z.value)

    np.testing.assert_equal([x.value, y.value, z.value],
            u.Quantity(p.cartesian).value)


def test_from_to_geodetic():
    lat, lon, height = p.geodetic(ell)
    b_p = Position3D.from_geodetic(lat, lon, height, ell=ell)
    np.testing.assert_array_almost_equal(u.Quantity(b_p.cartesian).value,
                                         [x.value, y.value, z.value], decimal=5)


def test_from_to_spherical():
    lat, lon, radius = p.spherical()
    b_p = Position3D.from_spherical(lat, lon, radius)
    np.testing.assert_array_almost_equal(u.Quantity(b_p.cartesian).value,
                                         [x.value, y.value, z.value],
                                         decimal=5)


def test_from_to_ellipsoidal():
    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    p = Position3D(x_, y_, z_)
    rlat, lon, u_ax = p.ellipsoidal(ell=ell)
    b_x, b_y, b_z = Position3D.from_ellipsoidal(rlat, lon, u_ax,
            ell=ell).cartesian
    np.testing.assert_array_almost_equal([b_x.value, b_y.value, b_z.value],
                                         [x_.value, y_.value, z_.value], decimal=5)

def test_to_enu_ell():
    lat0, lon0, height0 = 55.0*u.deg, 37.0*u.deg, 100.0 *u.m
    origin = (lat0, lon0, height0)
    enu = p.enu(origin=origin, ell=ell)

    b_x, b_y, b_z = enu_to_ecef(*enu, origin=origin, ell=ell)

    x_, y_, z_ = p.cartesian

    np.testing.assert_array_almost_equal(b_x.value, x_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z_.value, decimal=5)
