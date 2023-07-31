
import astropy.units as u
import numpy as np
import pytest
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.frame import ECEF
from pygeoid.coordinates.transform import enu_to_ecef

ell = Ellipsoid('GRS80')

# test data
n_test = 10  # * 2
r_ = np.geomspace(1, 1e8, num=n_test)
r_ = np.append(-r_[::-1], r_)
x, y, z = np.meshgrid(r_, r_, r_, indexing='ij') * u.m
p = ECEF(x, y, z)


def test_cartesian():
    np.testing.assert_equal(x.value, p.x.value)
    np.testing.assert_equal(y.value, p.y.value)
    np.testing.assert_equal(z.value, p.z.value)

    np.testing.assert_equal([x.value, y.value, z.value],
            p.cartesian.xyz.value)


def test_from_to_geodetic():
    b_p = ECEF.from_geodetic(p.geodetic.lat,
            p.geodetic.lon, p.geodetic.height, ell=ell)
    np.testing.assert_array_almost_equal(b_p.cartesian.xyz.value,
                                         [x.value, y.value, z.value], decimal=5)


def test_from_to_spherical():
    b_p = ECEF.from_spherical(p.spherical.lat, p.spherical.lon,
            p.spherical.distance)
    np.testing.assert_array_almost_equal(b_p.cartesian.xyz.value,
                                         [x.value, y.value, z.value],
                                         decimal=5)


def test_from_to_ellipsoidal():
    cond = (x**2 + y**2 + z**2) < ell.linear_eccentricity**2

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    p = ECEF(x_, y_, z_)
    b_x, b_y, b_z = ECEF.from_ellipsoidal_harmonic(p.ellipsoidal_harmonic.rlat,
            p.ellipsoidal_harmonic.lon, p.ellipsoidal_harmonic.u_ax,
            ell=ell).cartesian.xyz.value
    np.testing.assert_array_almost_equal([b_x, b_y, b_z],
                                         [x_.value, y_.value, z_.value], decimal=5)

def test_to_enu_ell():
    lat0, lon0, height0 = 55.0*u.deg, 37.0*u.deg, 100.0 *u.m
    origin = (lat0, lon0, height0)
    enu = p.enu(origin=origin, ell=ell)

    b_x, b_y, b_z = enu_to_ecef(*enu, origin=origin, ell=ell)

    x_, y_, z_ = p.cartesian.xyz

    np.testing.assert_array_almost_equal(b_x.value, x_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_y.value, y_.value, decimal=5)
    np.testing.assert_array_almost_equal(b_z.value, z_.value, decimal=5)
