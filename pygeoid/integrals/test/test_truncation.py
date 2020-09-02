
import pytest
import numpy as np

from pygeoid.integrals.truncation import molodensky_truncation_coefficients as Qn
from pygeoid.integrals.truncation import paul_coefficients

n = 60


def test_check_params():

    with pytest.raises(ValueError):
        Qn(0.5, 5, method='wrong_method')

    with pytest.raises(ValueError):
        Qn(190, 5)

    with pytest.raises(ValueError):
        Qn(90, -5)


def test_qn():
    """Test Molodensky truncation coefficients.

    """

    psi = np.linspace(0, 180, 19, endpoint=True)

    for psi0 in psi:
        # Hagiwara
        qn_hgwr = Qn(psi0, n, method='hagiwara')
        # Numerical
        qn_num = Qn(psi0, n, method='numerical')

        np.testing.assert_almost_equal(
            qn_hgwr, qn_num, 10,
            err_msg='Qn for psi = ' + str(psi0) + ' is failed!')

def test_0_and_180():

    n = 100
    # all coeffs for psi=180 is zero
    np.testing.assert_array_equal(
            Qn(180, n), 0, 10)

    # coeffs for psi=0 is 2 / (n - 1)
    q = np.array([2 / (i - 1) if i > 1 else 0 for i in range(n + 1)])
    np.testing.assert_array_equal(
            Qn(0, n), q)

    # numbers from Paul (1983)
    np.testing.assert_almost_equal(
            Qn(60, 1),
            np.array([-0.7611, -0.8543]), 4)


def test_paul_coefficients():
    """Test Paul coefficients.

    """
    psi = np.linspace(0, 180, 3, endpoint=True)

    for psi0 in psi:
        # Paul
        rnk_paul = paul_coefficients(psi0, n, method='paul')

        # Numerical
        rnk_num = paul_coefficients(psi0, n, method='numerical')

        np.testing.assert_almost_equal(
            rnk_paul, rnk_num, 10,
            err_msg='Rnk for psi = ' + str(psi0) + ' is failed!')
