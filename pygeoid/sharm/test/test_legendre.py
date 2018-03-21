
import numpy as np
from pygeoid.sharm import legendre


def test_lplm_lplm_d_equality():
    l_max = 2190
    x_test = np.linspace(-1, 1, 10)

    for x in x_test:
        lpmn_1 = legendre.lplm_d1(l_max, x)[0]
        lpmn_2 = legendre.lplm(l_max, x)

        np.testing.assert_almost_equal(
            lpmn_1, lpmn_2, decimal=16)


def test_lplm_d_None():
    l_max = 2190
    x = [-1, -0.999999, 0.999999, 1]
    lpmn_d = [legendre.lplm_d1(l_max, xi)[1] for xi in x]
    np.testing.assert_array_equal(lpmn_d, None)
