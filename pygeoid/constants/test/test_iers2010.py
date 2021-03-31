
import numpy as np
import astropy.units as u
import pygeoid.constants.iers2010 as iers2010


def test_tcg_to_tt():

    x_tcg = iers2010.GM_earth_tcg
    x_tt = iers2010.tcg_to_tt(x_tcg)
    np.testing.assert_almost_equal(x_tt.value / 10e5,
            iers2010.GM_earth_tt.value / 10e5, 0)


def test_l2_shida_number():
    np.testing.assert_almost_equal(
            iers2010.l2_shida_number().value, iers2010.l2.value, 4)

    np.testing.assert_almost_equal(
            iers2010.l2_shida_number(lat=0*u.deg).value,
            iers2010.l2.value - 0.0002 / 2, 4)

    np.testing.assert_almost_equal(
            iers2010.l2_shida_number(lat=90*u.deg).value,
            iers2010.l2.value + 0.0002, 4)

def test_h2_love_number():
    np.testing.assert_almost_equal(
            iers2010.h2_love_number().value, iers2010.h2.value, 4)

    np.testing.assert_almost_equal(
            iers2010.h2_love_number(lat=0*u.deg).value,
            iers2010.h2.value + 0.0006 / 2, 4)

    np.testing.assert_almost_equal(
            iers2010.h2_love_number(lat=90*u.deg).value,
            iers2010.h2.value - 0.0006, 4)


