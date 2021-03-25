
import pytest
import numpy as np
import astropy.units as u

from pygeoid.integrals.stokes import StokesKernel


def test_stokes_kernel():
    """Test Stokes function.

    """
    psi = np.linspace(0.01, 180, endpoint=False, dtype=np.float128) * u.deg
    t = np.cos(psi)

    st = StokesKernel()
    # Spherical distance
    st_psi = st.kernel(psi)

    # Stokes kernel for t = cos(psi)
    st_t = st._kernel_t(t)

    # Compare spherical distance and parameter t
    np.testing.assert_almost_equal(st_psi.value, st_t.value)


def test_stokes_kernel_derivatives():
    """Test Stokes function derivatives.

    """
    psi = np.linspace(0.1, 180, endpoint=False, dtype=np.float128) * u.deg
    t = np.cos(psi)

    st = StokesKernel()
    # Spherical distance
    st_psi = st.derivative_spherical_distance(psi)

    # Stokes kernel for t = cos(psi)
    st_t = st._derivative_t(t)

    # Compare spherical distance and parameter t
    # dS / dt = (dS / dpsi) * (dpsi / dt)
    np.testing.assert_almost_equal((st_psi / (-np.sqrt(1 - t**2))).value,
            st_t.value, 5)


def test_t_parameter():

    with pytest.raises(ValueError):
        StokesKernel()._kernel_t(-10)

    with pytest.raises(ValueError):
        StokesKernel()._kernel_t(10)

    with pytest.raises(ValueError):
        StokesKernel()._derivative_t(-10)

    with pytest.raises(ValueError):
        StokesKernel()._derivative_t(10)


