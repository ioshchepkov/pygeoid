
import pytest
import numpy as np

from pygeoid.integrals.stokes import StokesKernel


def test_stokes_kernel():
    """Test Stokes function.

    """
    psi = np.linspace(0.01, 180, endpoint=True, dtype=np.float128)
    psi_rad = np.radians(psi)
    t = np.cos(psi_rad)

    st = StokesKernel()
    # Spherical distance
    st_psi = st.kernel(psi, degrees=True)

    # Spherical distance in radians
    st_psi_rad = st.kernel(psi_rad, degrees=False)

    # Compare spherical distance in degrees amd radians
    np.testing.assert_almost_equal(st_psi, st_psi_rad)

    # Stokes kernel for t = cos(psi)
    st_t = st._kernel_t(t)

    # Compare spherical distance and parameter t
    np.testing.assert_almost_equal(st_psi, st_t)


def test_stokes_kernel_derivatives():
    """Test Stokes function derivatives.

    """
    psi = np.linspace(0.1, 180, endpoint=False, dtype=np.float128)
    psi_rad = np.radians(psi)
    t = np.cos(psi_rad)

    st = StokesKernel()
    # Spherical distance
    st_psi = st.derivative_spherical_distance(psi, degrees=True)

    # Spherical distance in radians
    st_psi_rad = st.derivative_spherical_distance(psi_rad, degrees=False)

    # Compare spherical distance in degrees amd radians
    np.testing.assert_almost_equal(st_psi, st_psi_rad)

    # Stokes kernel for t = cos(psi)
    st_t = st._derivative_t(t)

    # Compare spherical distance and parameter t
    # dS / dt = (dS / dpsi) * (dpsi / dt)
    np.testing.assert_almost_equal(st_psi / (-np.sqrt(1 - t**2)), st_t, 5)


def test_t_parameter():

    with pytest.raises(ValueError):
        StokesKernel()._kernel_t(-10)

    with pytest.raises(ValueError):
        StokesKernel()._kernel_t(10)

    with pytest.raises(ValueError):
        StokesKernel()._derivative_t(-10)

    with pytest.raises(ValueError):
        StokesKernel()._derivative_t(10)


