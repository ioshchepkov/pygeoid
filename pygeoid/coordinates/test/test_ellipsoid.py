
import numpy as np
from pygeoid.coordinates.ellipsoid import Ellipsoid

ell = Ellipsoid('GRS80')

def test_short_long_names():
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
    assert ell.a ==  6378137.0
    assert np.allclose(ell.b, 6356752.3141)
    assert np.allclose(ell.linear_eccentricity, 521854.0097)
    assert np.allclose(ell.polar_curvature_radius, 6399593.6259)
    assert np.allclose(ell.eccentricity_squared, 0.00669438002290)
    assert np.allclose(ell.second_eccentricity_squared,
            0.00673949677548)
    assert np.allclose(ell.flattening, 0.00335281068118)
    assert np.allclose(ell.reciprocal_flattening, 298.257222101)
    assert np.allclose(ell.quadrant_distance, 10001965.7293)
    assert np.allclose(ell.surface_area, 5.10065622e+014)
    assert np.allclose(ell.volume, 1.08320732e+021)

    # mean radius
    kind = 'arithmetic'
    assert np.allclose(ell.mean_radius(kind), 6371008.7714)
    kind = 'same_area'
    assert np.allclose(ell.mean_radius(kind), 6371007.1810)
    kind = 'same_volume'
    assert np.allclose(ell.mean_radius(kind), 6371000.7900)

    ##########################################################################
    # Latitude dependend values
    # Deakin, R.E. and Hunter, M.N., 2010. Geometric geodesy part A. Lecture
    # Notes, School of Mathematical & Geospatial Sciences, RMIT University,
    # Melbourne, Australia.
    # pp. 87 - 88
    lat = -(37 + 48/60 + 33.1234 / 3600)
    rlat = np.radians(lat)
    assert np.allclose(ell._w(rlat), 0.998741298)
    assert np.allclose(ell._v(rlat), 1.002101154)
    assert np.allclose(ell.meridian_curvature_radius(rlat), 6359422.962)
    assert np.allclose(ell.prime_vertical_curvature_radius(rlat), 6386175.289)
    assert np.allclose(ell.mean_curvature(lat), 1/6372785.088)
    assert np.allclose(ell.meridian_arc_distance(0.0, lat), 4186320.340)
    ##########################################################################

    # forward and inverse geodetic problems
    # TODO: tests for fwd and inv
    assert np.allclose(ell.meridian_arc_distance(0.0, 90),
            ell.quadrant_distance)
    circle_length = 2*np.pi*ell.a
    assert np.allclose(ell.parallel_arc_distance(0.0, 0.0, 90.),
            circle_length/4)

    # TODO: tests for radiuses
    # TODO: tests for latitudes
