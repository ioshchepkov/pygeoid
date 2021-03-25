
import os
import pytest
import numpy as np
import astropy.units as u
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
    np.testing.assert_equal(ell.a.value, 6378137.0)
    np.testing.assert_almost_equal(ell.b.value, 6356752.3141, decimal=4)
    np.testing.assert_almost_equal(ell.linear_eccentricity.value,
                                   521854.0097, decimal=4)
    np.testing.assert_almost_equal(ell.polar_curvature_radius.value,
                                   6399593.6259, decimal=4)
    np.testing.assert_almost_equal(ell.eccentricity_squared.value,
                                   0.00669438002290,
                                   decimal=14)
    np.testing.assert_almost_equal(ell.second_eccentricity_squared.value,
                                   0.00673949677548,
                                   decimal=14)
    np.testing.assert_almost_equal(ell.flattening.value,
                                   0.00335281068118,
                                   decimal=14)
    np.testing.assert_almost_equal(ell.reciprocal_flattening.value,
                                   298.257222101,
                                   decimal=9)
    np.testing.assert_almost_equal(ell.quadrant_distance.value,
                                   10001965.7293, decimal=4)
    np.testing.assert_almost_equal(ell.surface_area.value * 1e-014,
                                   5.10065622, decimal=8)
    np.testing.assert_almost_equal(ell.volume.value * 1e-021,
                                   1.08320732, decimal=8)

    # mean radius
    kind = 'arithmetic'
    np.testing.assert_almost_equal(
        ell.mean_radius(kind).value, 6371008.7714, decimal=4)
    kind = 'same_area'
    np.testing.assert_almost_equal(ell.mean_radius(kind).value,
                                   6371007.1810, decimal=4)
    kind = 'same_volume'
    np.testing.assert_almost_equal(ell.mean_radius(kind).value,
                                   6371000.7900, decimal=4)


def test_latitude_dependend_values():
    # From
    # Deakin, R.E. and Hunter, M.N., 2010. Geometric geodesy part A. Lecture
    # Notes, School of Mathematical & Geospatial Sciences, RMIT University,
    # Melbourne, Australia.
    # pp. 87 - 88
    ell = Ellipsoid('GRS80')
    lat = -(37 + 48 / 60 + 33.1234 / 3600) * u.deg

    np.testing.assert_almost_equal(ell._w(lat).value, 0.998741298, decimal=9)
    np.testing.assert_almost_equal(ell._v(lat).value, 1.002101154, decimal=9)
    np.testing.assert_almost_equal(ell.meridian_curvature_radius(lat).value,
                                   6359422.962, decimal=3)
    np.testing.assert_almost_equal(ell.prime_vertical_curvature_radius(lat).value,
                                   6386175.289, decimal=3)
    np.testing.assert_almost_equal(ell.mean_curvature(lat).value,
                                   1 / 6372785.088, decimal=16)
    np.testing.assert_almost_equal(ell.meridian_arc_distance(0.0 * u.deg, lat).value,
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
    b_lat2, b_lon2, b_azi2 = ell.fwd(
            lat1*u.deg, lon1*u.deg, azi1*u.deg, dist12*u.m)

    np.testing.assert_array_almost_equal(
            b_lat2.to('degree').value, lat2, decimal=8)
    np.testing.assert_array_almost_equal(
            b_lon2.to('degree').value, lon2, decimal=8)
    np.testing.assert_array_almost_equal(
            b_azi2.to('degree').value, azi2 - 180., decimal=8)


def test_inv():
    ell = Ellipsoid('WGS84')

    # check meridian arc
    np.testing.assert_almost_equal(ell.meridian_arc_distance(
        0.0*u.deg, 90*u.deg).value, ell.quadrant_distance.value, decimal=4)

    # check equatorial circle
    circle_length = 2 * np.pi * ell.a
    np.testing.assert_almost_equal(ell.parallel_arc_distance(0.0*u.deg,
        0.0*u.deg, 90. *u.deg).value, circle_length.value / 4, decimal=5)

    # equator = geodesic
    np.testing.assert_almost_equal(ell.parallel_arc_distance(0.0*u.deg,
        0.0*u.deg, 90.*u.deg).value, ell.inv(0.0 * u.deg, 0.0 *u.deg, 0.0*u.deg,
                                       90.0*u.deg)[-1].value, decimal=5)

    # Load short test data from Geographiclib
    # https://geographiclib.sourceforge.io/html/geodesic.html#testgeod
    path = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(path, 'data/GeodTest-short.dat.gz')
    (lat1, lon1, azi1, lat2, lon2, azi2, dist12) = np.loadtxt(
        fname, usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True, dtype=np.float64)

    b_azi1, b_azi2, b_dist12 = ell.inv(
            lat1 * u.deg, lon1 * u.deg, lat2 * u.deg, lon2 * u.deg)

    # NOTE: Loss of precision
    np.testing.assert_array_almost_equal(
            b_azi1.to('degree').value, azi1, decimal=3)
    np.testing.assert_array_almost_equal(
            b_azi2.to('degree').value, azi2 - 180., decimal=3)
    np.testing.assert_array_almost_equal(
            b_dist12.to('m').value, dist12, decimal=8)


def test_radiuses():
    ell = Ellipsoid('GRS80')
    # geodetic latitude
    lat = np.arange(-90, 90, 0.1, dtype=float) * u.deg
    x, y, _ = geodetic_to_cartesian(lat, lon=0.0 * u.deg, height=0.0 * u.m, ell=ell)
    np.testing.assert_array_almost_equal(ell.circle_radius(lat).value,
                                         np.sqrt(x**2 + y**2).value, decimal=5)

    sph_lat, _, sph_radius = geodetic_to_spherical(lat, lon=0.0*u.deg,
                                                   height=0.0*u.m, ell=ell)

    np.testing.assert_array_almost_equal(ell.polar_equation(sph_lat).value,
                                         sph_radius.value, decimal=5)


def test_latitudes():
    ell = Ellipsoid('GRS80')
    lat = np.arange(-90, 90, 0.1, dtype=float) * u.deg

    sph_lat, _, sph_radius = geodetic_to_spherical(lat, lon=0.0 * u.deg,
                                                   height=0.0 * u.m, ell=ell)

    np.testing.assert_array_almost_equal(ell.geocentric_latitude(lat).to('degree').value,
                                         sph_lat.to('degree').value, decimal=8)

    reduced_latitude = np.degrees(
        np.arctan(np.tan(np.radians(sph_lat)) / (1 - ell.f)))

    np.testing.assert_array_almost_equal(ell.reduced_latitude(lat).to('degree').value,
                                         reduced_latitude.to('degree').value, decimal=8)

    # compare with series from
    # J. P. Snyder, Map projections - a working  manual, 1926, page 16
    s2lat = (ell.e2 / 3 + 31*ell.e2**2 / 180 +
             59*ell.e2**3 / 560)*np.sin(2*lat)
    s4lat = (17*ell.e2**2 / 360 + 61*ell.e2**3 / 1260) * np.sin(4*lat)
    s6lat = (383 * ell.e2**3 / 45360) * np.sin(6 * lat)

    authalic_latitude = lat.to('radian').value - s2lat.value + s4lat.value - s6lat.value

    np.testing.assert_array_almost_equal(
            ell.authalic_latitude(lat).to('degree').value,
            np.degrees(authalic_latitude), decimal=6)
