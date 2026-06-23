import numpy as np
from astropy.coordinates import CartesianDifferential, CartesianRepresentation

from pygeoid.conventions import units as u
from pygeoid.fields.operators import differential, gradient, hessian
from pygeoid.fields.potential.base import PotentialBase


class TestPotential(PotentialBase):
    def _potential(self, position):
        return position.x**2 / u.s**2

    def _differential(self, position):
        return CartesianDifferential(
            2 * position.x / u.s**2,
            np.zeros(position.shape) * u.m / u.s**2,
            np.zeros(position.shape) * u.m / u.s**2,
        )

    def _hessian(self, position):
        return np.eye(3)


def test_potential_is_scalar_field():
    field = TestPotential()
    position = CartesianRepresentation(3 * u.m, 0 * u.m, 0 * u.m)

    assert field.value(position) == field.potential(position)


def test_field_operator_wrappers():
    field = TestPotential()
    position = CartesianRepresentation(3 * u.m, 0 * u.m, 0 * u.m)

    np.testing.assert_allclose(differential(field, position).d_x.value, 6)
    np.testing.assert_allclose(gradient(field, position).value, 6)
    np.testing.assert_allclose(hessian(field, position), np.eye(3))
