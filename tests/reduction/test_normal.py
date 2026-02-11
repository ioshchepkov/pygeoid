
#import os
import pytest
import numpy as np
import astropy.units as u
import pygeoid.coordinates.transform as transform
from pygeoid.reduction.normal import LevelEllipsoid
from pygeoid.reduction.normal import surface_normal_gravity_clairaut


def test_init():
    with pytest.raises(ValueError):
        ell = LevelEllipsoid('xxx')


def test_short_long_names():
    ell = LevelEllipsoid()
    assert ell._surface_potential == ell.surface_potential
    assert ell._gamma_e == ell.equatorial_normal_gravity
    assert ell._gamma_p == ell.polar_normal_gravity
    assert ell._gravity_flattening == ell.gravity_flattening
    assert ell._j2 == ell.j2

def test_against_GRS80():
    # Moritz, H., 1980. Geodetic reference system 1980.
    # Bulletin géodésique, 54(3), pp.395-405.
    ell = LevelEllipsoid('GRS80')
    # geometric constants
    np.testing.assert_equal(ell.a.value, 6378137.0)
    np.testing.assert_almost_equal(ell.b.value, 6356752.3141, decimal=4)
    np.testing.assert_almost_equal(ell.linear_eccentricity.value,
                                   521854.0097, decimal=4)
    np.testing.assert_almost_equal(ell.polar_curvature_radius.value,
                                   6399593.6259, decimal=4)
    np.testing.assert_almost_equal(ell.eccentricity_squared.value,
                                   0.00669438002290, decimal=14)
    np.testing.assert_almost_equal(ell.second_eccentricity_squared.value,
                                   0.00673949677548, decimal=14)
    np.testing.assert_almost_equal(ell.flattening.value,
                                   0.00335281068118, decimal=14)
    np.testing.assert_almost_equal(ell.reciprocal_flattening.value,
                                   298.257222101, decimal=9)
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

    # physical constants
    np.testing.assert_almost_equal(ell.surface_potential.value,
                                   62636860.850, decimal=3)
    np.testing.assert_almost_equal(ell.equatorial_normal_gravity.value,
                                   9.7803267715, decimal=10)
    np.testing.assert_almost_equal(ell.polar_normal_gravity.value,
                                   9.8321863685, decimal=10)
    np.testing.assert_almost_equal(ell.gravity_flattening.value,
                                   0.005302440112, decimal=12)
    np.testing.assert_almost_equal(ell._k.value,
                                   0.001931851353, decimal=12)

    np.testing.assert_almost_equal(ell.m.value,
                                   0.00344978600308, decimal=14)
    np.testing.assert_almost_equal(ell.j2n(n=2).value,
                                   -0.00000237091222, decimal=14)
    np.testing.assert_almost_equal(ell.j2n(n=3).value,
                                   0.00000000608347, decimal=14)
    np.testing.assert_almost_equal(ell.j2n(n=4).value,
                                   -0.00000000001427, decimal=14)

    np.testing.assert_almost_equal(ell.surface_normal_gravity(45. * u.deg).value,
                                   9.806199203, decimal=9)

    np.testing.assert_almost_equal(ell.conventional_gravity_coeffs()[-1].value,
                                   0.0000058, decimal=7)

    # other
    np.testing.assert_almost_equal(ell.mean_normal_gravity.value,
                                   9.797644656, decimal=9)
    np.testing.assert_almost_equal(
        ell.surface_normal_gravity(45. * u.deg).value,
        9.806199203, decimal=9)


def test_surface_gravity_potential():
    ell = LevelEllipsoid('GRS80')
    # geodetic latitude
    lat = np.arange(-90, 90, 0.1, dtype=float) * u.deg
    rlat, _, u_ax = transform.geodetic_to_ellipsoidal(lat=lat,
                                                   lon=0.0*u.deg,
                                                   height=0.0*u.m, ell=ell)
    gravity_potential_0 = ell.gravity_potential(rlat, u_ax)
    np.testing.assert_almost_equal(
        ell.surface_potential.value,
        gravity_potential_0.value, decimal=5)
    geoclat, _, radius = transform.geodetic_to_spherical(lat=lat,
                                                         lon=0.0*u.deg,
                                                         height=0.0*u.m, ell=ell)
    gravity_potential_sph_0 = ell.gravity_potential_sph(geoclat, radius)
    np.testing.assert_almost_equal(
        ell.surface_potential.value,
        gravity_potential_sph_0.value, decimal=5)
    np.testing.assert_almost_equal(
        gravity_potential_sph_0.value,
        gravity_potential_0.value, decimal=5)


def test_potential():
    ell = LevelEllipsoid('GRS80')

    n_test = 100  # * 2
    r_ = np.geomspace(ell.b.value, 1e8, num=n_test)
    r_ = np.append(-r_[::-1], r_)
    x, y, z = np.meshgrid(r_, r_, r_, indexing='ij') * u.m

    sphlat, _, radius = transform.cartesian_to_spherical(x, y, z)
    cond = radius < ell.polar_equation(sphlat)

    x_ = np.ma.masked_where(cond, x).compressed()
    y_ = np.ma.masked_where(cond, y).compressed()
    z_ = np.ma.masked_where(cond, z).compressed()

    sphlat, _, radius = transform.cartesian_to_spherical(x_, y_, z_)
    rlat, _, u_ax = transform.cartesian_to_ellipsoidal(x_, y_, z_, ell=ell)
    gravity_potential_ell = ell.gravity_potential(rlat, u_ax)
    gravity_potential_sph = ell.gravity_potential_sph(sphlat, radius)

    np.testing.assert_almost_equal(
        gravity_potential_sph.value,
        gravity_potential_ell.value, decimal=4)

    gravitational_potential_ell = ell.gravitational_potential(rlat, u_ax)
    gravitational_potential_sph = ell.gravitational_potential_sph(
        sphlat, radius)

    np.testing.assert_almost_equal(
        gravitational_potential_sph.value,
        gravitational_potential_ell.value, decimal=4)


def test_surface_normal_gravity():
    ell = LevelEllipsoid('GRS80')
    np.testing.assert_almost_equal(
        ell.surface_normal_gravity(0. * u.deg).value,
        ell.equatorial_normal_gravity.value, decimal=9)
    np.testing.assert_almost_equal(
        ell.surface_normal_gravity(90. * u.deg).value,
        ell.polar_normal_gravity.value, decimal=9)

    # geodetic latitude
    lat = np.arange(-90, 90, 0.1, dtype=float) * u.deg
    rlat, _, u_ax = transform.geodetic_to_ellipsoidal(lat=lat,
                                                   lon=0.0 * u.deg,
                                                   height=0.0 * u.m, ell=ell)

    normal_gravity_somigliana = ell.surface_normal_gravity(lat)
    normal_gravity_ell = ell.normal_gravity(rlat, u_ax)

    np.testing.assert_almost_equal(
        normal_gravity_somigliana.value,
        normal_gravity_ell.value,
        decimal=9)

    # approximate formula (Clairaut)
    beta, beta1 = ell.conventional_gravity_coeffs()
    ge = ell.equatorial_normal_gravity
    latrad = np.radians(lat)
    normal_gravity_approx = ge*(1 + beta*np.sin(latrad)**2 -
                                beta1*np.sin(2*latrad)**2)

    np.testing.assert_almost_equal(
        normal_gravity_somigliana.value,
        normal_gravity_approx.value,
        decimal=6)

    np.testing.assert_almost_equal(
        surface_normal_gravity_clairaut(lat, '1980').to('mGal').value,
        normal_gravity_approx.to('mGal').value,
        decimal=1)


def test_normal_gravity_gradient():
    ell = LevelEllipsoid('GRS80')
    # geodetic latitude
    lat = np.arange(-90, 90, 0.5, dtype=float) * u.deg
    grad_ell_0 = ell.surface_vertical_normal_gravity_gradient(lat)
    np.testing.assert_almost_equal(
        grad_ell_0.value, -0.3086e-5,
        decimal=8)


def test_normal_gravity():
    ell = LevelEllipsoid('GRS80')
    # geodetic latitude
    lat = np.arange(-90, 90, 0.5, dtype=np.float64) * u.deg
    height = np.arange(1, 10e3, 100, dtype=np.float64) * u.m
    latv, heightv = np.meshgrid(lat, height, indexing='xy')

    height_corr = ell.height_correction(latv, heightv)
    normal_gravity_somigliana = ell.surface_normal_gravity(latv)
    normal_gravity_1 = normal_gravity_somigliana + height_corr

    rlat, _, u_ax = transform.geodetic_to_ellipsoidal(lat=latv,
                                                   lon=0.0*u.deg, height=heightv, ell=ell)
    normal_gravity_2 = ell.normal_gravity(rlat, u_ax)

    np.testing.assert_almost_equal(
        normal_gravity_1.value,
        normal_gravity_2.value,
        decimal=6)
