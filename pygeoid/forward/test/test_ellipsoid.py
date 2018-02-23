
#import os
import pytest
import numpy as np
from pygeoid.forward.ellipsoid import LevelEllipsoid
# from pygeoid.coordinates.transform import (geodetic_to_cartesian,
#                                           geodetic_to_spherical)


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

    # physical constants
    np.testing.assert_almost_equal(ell.surface_potential,
                                   62636860.850, decimal=3)
    np.testing.assert_almost_equal(ell.equatorial_normal_gravity,
                                   9.7803267715, decimal=10)
    np.testing.assert_almost_equal(ell.polar_normal_gravity,
                                   9.8321863685, decimal=10)
    np.testing.assert_almost_equal(ell.gravity_flattening,
                                   0.005302440112, decimal=12)
    np.testing.assert_almost_equal(ell._k,
                                   0.001931851353, decimal=12)

    np.testing.assert_almost_equal(ell.m,
                                   0.00344978600308, decimal=14)
    np.testing.assert_almost_equal(ell.j2n(n=2),
                                   -0.00000237091222, decimal=14)
    np.testing.assert_almost_equal(ell.j2n(n=3),
                                   0.00000000608347, decimal=14)
    np.testing.assert_almost_equal(ell.j2n(n=4),
                                   -0.00000000001427, decimal=14)

    np.testing.assert_almost_equal(ell.surface_normal_gravity(45.),
                                   9.806199203, decimal=9)
