
import numpy as np
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.transform import *

ell = Ellipsoid('GRS80')

# test cartesian data
r_max = 100000e3 # km * 10**3 = m
x = np.around(np.linspace(-r_max, r_max, 100000, dtype=np.float64), 9)
y = np.around(np.linspace(-r_max, r_max, 100000, dtype=np.float64), 9)
z = np.around(np.linspace(-r_max, r_max, 100000, dtype=np.float64), 9)


def test_cartesian_to_geodetic_and_back():
    #lat, lon, height = cartesian_to_geodetic(x, y, z, ell)
    #b_x, b_y, b_z = geodetic_to_cartesian(lat, lon, height, ell)

    #assert np.allclose(x, b_x)
    #assert np.allclose(y, b_y)
    #assert np.allclose(z, b_z)

    #lat, lon, height = cartesian_to_geodetic(100, 200, 1000, ell)
    #b_x, b_y, b_z = geodetic_to_cartesian(lat, lon, height, ell)
    #assert np.allclose(100, b_x)
    #assert np.allclose(200, b_y)
    #assert np.allclose(1000, b_z)
    pass


def test_cartesian_to_spherical_and_back():
    lat, lon, radius = cartesian_to_spherical(x, y, z)
    b_x, b_y, b_z = spherical_to_cartesian(lat, lon, radius)

    assert np.allclose(x, b_x)
    assert np.allclose(y, b_y)
    assert np.allclose(z, b_z)


def test_cartesian_to_ellipsoidal_and_back():
    mask = np.where((x**2 + y**2 + z**2) >= ell.linear_eccentricity**2)[0]

    rlat, lon, u = cartesian_to_ellipsoidal(x[mask], y[mask], z[mask], ell)
    b_x, b_y, b_z = ellipsoidal_to_cartesian(rlat, lon, u, ell)

    assert np.allclose(x[mask], b_x)
    assert np.allclose(y[mask], b_y)
    assert np.allclose(z[mask], b_z)
