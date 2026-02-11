
import os
import numpy as np
import pygeoid.reduction.atmosphere as atm
import astropy.units as u


def test_ussa1976_density():
    fname = os.path.join(os.path.dirname(__file__),
                         'data/ussa1976_density.txt.gz')
    test_heights, test_density = np.loadtxt(fname, unpack=True,
                                            delimiter='\t', skiprows=6, dtype=float)

    test_heights *= u.m
    test_density *= (u.kg / u.m**3)

    density = atm.ussa76_density(test_heights)
    np.testing.assert_almost_equal(density.to('kg / m3').value,
            test_density.value, decimal=5)


def test_atm_corr():
    n_samples = 50
    h_min = 0.0
    h_max = 10000
    decimal = 2
    test_heights = np.linspace(h_min, h_max, n_samples) * u.m

    grs80 = atm.grs80_atm_corr_interp(test_heights).to('mGal').value
    pz90 = atm.pz90_atm_corr(test_heights).to('mGal').value
    wenz = atm.wenzel_atm_corr(test_heights).to('mGal').value

    iag = np.array([atm.iag_atm_corr_sph(atm.ussa76_density, hi, 84852*u.m,
        n_samples).to('mGal').value for hi in test_heights])

    np.testing.assert_almost_equal(grs80, pz90, decimal=decimal)
    np.testing.assert_almost_equal(grs80, wenz, decimal=decimal)
    np.testing.assert_almost_equal(grs80, iag, decimal=decimal)
