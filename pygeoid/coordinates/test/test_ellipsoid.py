
import os
import pytest
import numpy as np
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.transform import (geodetic_to_cartesian,
                                           geodetic_to_spherical)


def test_init():
    with pytest.raises(ValueError):
        ell = Ellipsoid('xxx')


def test_short_long_names():
    ell = Ellipsoid()
    assert ell.a == ell.equatorial_radius
    assert ell.b == ell.polar_radius
    assert ell.f == ell.flattening
    assert ell.e2 == ell.eccentricity_squared
    assert ell.e == ell.eccentricity
    assert ell.e12 == ell.second_eccentricity_squared
    assert ell.e1 == ell.second_eccentricity


def test_against_GRS80():
    # Moritz, H., 1980. Geodetic reference system 1980.
    # Bulletin géodésique, 54(3), pp.395-405.
    ell = Ellipsoid('GRS80')
    np.testing.assert_equal(ell.a, 6378137.0)
    np.testing.assert_almost_equal(ell.b, 6356752.3141, decimal=4)
    np.testing.assert_almost_equal(ell.linear_eccentricity,
                                   521854.0097, decimal=4)
    np.testing.assert_almost_equal(ell.polar_curvature_radius,
                                   6399593.6259, decimal=4)
    np.testing.assert_almost_equal(ell.eccentricity_squared,
                                   0.00669438002290, decimal=14)
    np.testing.assert_almost_equal(ell.second_eccentricity_squared,
                                   0.00673949677548, decimal=14)
    np.testing.assert_almost_equal(ell.flattening,
                                   0.00335281068118, decimal=14)
    np.testing.assert_almost_equal(ell.reciprocal_flattening,
                                   298.257222101, decimal=9)
    np.testing.assert_almost_equal(ell.quadrant_distance,
                                   10001965.7293, decimal=4)
    np.testing.assert_almost_equal(ell.surface_area * 1e-014,
                                   5.10065622, decimal=8)
    np.testing.assert_almost_equal(ell.volume * 1e-021,
                                   1.08320732, decimal=8)

    # mean radius
    kind = 'arithmetic'
    np.testing.assert_almost_equal(
        ell.mean_radius(kind), 6371008.7714, decimal=4)
    kind = 'same_area'
    np.testing.assert_almost_equal(ell.mean_radius(kind),
                                   6371007.1810, decimal=4)
    kind = 'same_volume'
    np.testing.assert_almost_equal(ell.mean_radius(kind),
                                   6371000.7900, decimal=4)


def test_latitude_dependend_values():
    # From
    # Deakin, R.E. and Hunter, M.N., 2010. Geometric geodesy part A. Lecture
    # Notes, School of Mathematical & Geospatial Sciences, RMIT University,
    # Melbourne, Australia.
    # pp. 87 - 88
    ell = Ellipsoid('GRS80')
    lat = -(37 + 48 / 60 + 33.1234 / 3600)
    rlat = np.radians(lat)

    np.testing.assert_almost_equal(ell._w(rlat), 0.998741298, decimal=9)
    np.testing.assert_almost_equal(ell._v(rlat), 1.002101154, decimal=9)
    np.testing.assert_almost_equal(ell.meridian_curvature_radius(rlat),
                                   6359422.962, decimal=3)
    np.testing.assert_almost_equal(ell.prime_vertical_curvature_radius(rlat),
                                   6386175.289, decimal=3)
    np.testing.assert_almost_equal(ell.mean_curvature(lat),
                                   1 / 6372785.088, decimal=16)
    np.testing.assert_almost_equal(ell.meridian_arc_distance(0.0, lat),
                                   4186320.340, decimal=3)


def test_fwd():
    ell = Ellipsoid('WGS84')
    # Load short test data from Geographiclib
    # https://geographiclib.sourceforge.io/html/geodesic.html#testgeod
    path = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(path, 'data/GeodTest-short.dat.gz')
    (lat1, lon1, azi1, lat2, lon2, azi2, dist12) = np.loadtxt(fname,
                                                              usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True, dtype=np.float64)

    # test forward
    b_lat2, b_lon2, b_azi2 = ell.fwd(lat1, lon1, azi1, dist12)

    np.testing.assert_array_almost_equal(b_lat2, lat2, decimal=8)
    np.testing.assert_array_almost_equal(b_lon2, lon2, decimal=8)
    np.testing.assert_array_almost_equal(b_azi2, azi2 - 180., decimal=8)


def test_inv():
    ell = Ellipsoid('WGS84')

    # check meridian arc
    np.testing.assert_almost_equal(ell.meridian_arc_distance(0.0, 90),
                                   ell.quadrant_distance, decimal=4)

    # check equatorial circle
    circle_length = 2 * np.pi * ell.a
    np.testing.assert_almost_equal(ell.parallel_arc_distance(0.0, 0.0, 90.),
                                   circle_length / 4, decimal=5)

    # Load short test data from Geographiclib
    # https://geographiclib.sourceforge.io/html/geodesic.html#testgeod
    path = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(path, 'data/GeodTest-short.dat.gz')
    (lat1, lon1, azi1, lat2, lon2, azi2, dist12) = np.loadtxt(
        fname, usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True, dtype=np.float64)

    b_azi1, b_azi2, b_dist12 = ell.inv(lat1, lon1, lat2, lon2)

    # NOTE: Loss of precision
    np.testing.assert_array_almost_equal(b_azi1, azi1, decimal=3)
    np.testing.assert_array_almost_equal(b_azi2, azi2 - 180., decimal=3)
    np.testing.assert_array_almost_equal(b_dist12, dist12, decimal=8)


def test_radiuses():
    ell = Ellipsoid('GRS80')
    # geodetic latitude
    lat = np.arange(-90, 90, 0.1, dtype=float)
    x, y, _ = geodetic_to_cartesian(lat, lon=0.0, height=0.0, ell=ell)
    np.testing.assert_array_almost_equal(ell.circle_radius(lat),
                                         np.sqrt(x**2 + y**2), decimal=5)

    sph_lat, _, sph_radius = geodetic_to_spherical(lat, lon=0.0,
                                                   height=0.0, ell=ell)

    np.testing.assert_array_almost_equal(ell.polar_equation(sph_lat),
                                         sph_radius, decimal=5)


def test_latitudes():
    ell = Ellipsoid('GRS80')
    lat = np.arange(-90, 90, 0.1, dtype=float)

    sph_lat, _, sph_radius = geodetic_to_spherical(lat, lon=0.0,
                                                   height=0.0, ell=ell)

    np.testing.assert_array_almost_equal(ell.geocentric_latitude(lat),
                                         sph_lat, decimal=8)

    reduced_latitude = np.degrees(
        np.arctan(np.tan(np.radians(sph_lat)) / (1 - ell.f)))

    np.testing.assert_array_almost_equal(ell.reduced_latitude(lat),
                                         reduced_latitude, decimal=8)

    # compare with series from
    # J. P. Snyder, Map projections - a working  manual, 1926, page 16
    latr = np.radians(lat)
    s2lat = (ell.e2 / 3 + 31*ell.e2**2 / 180 +
             59*ell.e2**3 / 560)*np.sin(2*latr)
    s4lat = (17*ell.e2**2 / 360 + 61*ell.e2**3 / 1260) * np.sin(4*latr)
    s6lat = (383 * ell.e2**3 / 45360) * np.sin(6 * latr)

    authalic_latitude = np.degrees(latr - s2lat + s4lat - s6lat)

    np.testing.assert_array_almost_equal(ell.authalic_latitude(lat),
                                         authalic_latitude, decimal=6)
