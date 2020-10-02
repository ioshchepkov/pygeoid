
import pytest
import numpy as np

from pygeoid.integrals.stokes import Stokes


def test_stokes_kernel():
    """Test Stokes function.

    """
    psi = np.linspace(0.01, 180, endpoint=True, dtype=np.float128)
    psi_rad = np.radians(psi)
    t = np.cos(psi_rad)

    # Spherical distance
    st_psi = Stokes.kernel(psi, degrees=True)

    # Spherical distance in radians
    st_psi_rad = Stokes.kernel(psi_rad, degrees=False)

    # Compare spherical distance in degrees amd radians
    np.testing.assert_almost_equal(st_psi, st_psi_rad)

    # Stokes kernel for t = cos(psi)
    st_t = Stokes._kernel_t(t)

    # Compare spherical distance and parameter t
    np.testing.assert_almost_equal(st_psi, st_t)


def test_stokes_kernel_derivatives():
    """Test Stokes function derivatives.

    """
    psi = np.linspace(0.1, 180, endpoint=False, dtype=np.float128)
    psi_rad = np.radians(psi)
    t = np.cos(psi_rad)

    # Spherical distance
    st_psi = Stokes.derivative_spherical_distance(psi, degrees=True)

    # Spherical distance in radians
    st_psi_rad = Stokes.derivative_spherical_distance(psi_rad, degrees=False)

    # Compare spherical distance in degrees amd radians
    np.testing.assert_almost_equal(st_psi, st_psi_rad)

    # Stokes kernel for t = cos(psi)
    st_t = Stokes._derivative_t(t)

    # Compare spherical distance and parameter t
    # dS / dt = (dS / dpsi) * (dpsi / dt)
    np.testing.assert_almost_equal(st_psi / (-np.sqrt(1 - t**2)), st_t, 5)


def test_t_parameter():

    with pytest.raises(ValueError):
        Stokes._kernel_t(-10)

    with pytest.raises(ValueError):
        Stokes._kernel_t(10)

    with pytest.raises(ValueError):
        Stokes._derivative_t(-10)

    with pytest.raises(ValueError):
        Stokes._derivative_t(10)


