import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, SphericalDifferential

from pygeoid.earth.gravity.centrifugal import Centrifugal

omega = 2.0 / u.s
potential = Centrifugal(omega=omega)
position = CartesianRepresentation(
    [1.0, 3.0] * u.m,
    [2.0, 4.0] * u.m,
    [5.0, 6.0] * u.m,
)


def test_potential_accepts_position():
    expected = 0.5 * omega**2 * (position.x**2 + position.y**2)

    np.testing.assert_allclose(potential.potential(position), expected)


def test_differential():
    differential = potential.differential(position)

    np.testing.assert_allclose(differential.d_x, omega**2 * position.x)
    np.testing.assert_allclose(differential.d_y, omega**2 * position.y)
    np.testing.assert_allclose(
        differential.d_z, np.zeros(position.shape) * u.m / u.s**2
    )


def test_differential_represents_as_requested_class():
    differential = potential.differential(position, SphericalDifferential)

    assert isinstance(differential, SphericalDifferential)


def test_gradient_accepts_position():
    expected = omega**2 * np.sqrt(position.x**2 + position.y**2)

    np.testing.assert_allclose(potential.gradient(position), expected)
