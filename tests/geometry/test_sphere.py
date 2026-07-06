import astropy.units as u
import numpy as np
import pytest

from pygeoid.geometry.surfaces import Sphere


def test_spherical_distance():
    distance = Sphere.spherical_distance(
        [0, 0] * u.deg,
        [0, 90] * u.deg,
        [0] * u.deg,
        [0] * u.deg,
    )

    np.testing.assert_allclose(distance.to_value(u.deg), [[0, 90]])


def test_check_spherical_distance():
    distance = Sphere.check_spherical_distance([0, np.pi] * u.rad)

    np.testing.assert_allclose(distance.to_value(u.deg), [0, 180])


@pytest.mark.parametrize("distance", [-1, 181])
def test_check_spherical_distance_rejects_out_of_range_values(distance):
    with pytest.raises(ValueError, match=r"within \[0°, 180°\]"):
        Sphere.check_spherical_distance(distance * u.deg)
