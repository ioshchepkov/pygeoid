
import os
import numpy as np
import pygeoid.reduction.atmosphere as atm


def test_ussa1976_density():
    fname = os.path.join(os.path.dirname(__file__),
                         'data/ussa1976_density.txt.gz')
    test_heights, test_density = np.loadtxt(fname, unpack=True,
                                            delimiter='\t', skiprows=6, dtype=float)

    density = atm.ussa76_density(test_heights)
    np.testing.assert_almost_equal(density, test_density, decimal=5)


def test_atm_corr():
    n_samples = 250
    h_min = 0.0
    h_max = 10000
    decimal = 2
    test_heights = np.linspace(h_min, h_max, n_samples)

    grs80 = atm.grs80_atm_corr_interp(test_heights)
    pz90 = atm.pz90_atm_corr(test_heights)
    wenz = atm.wenzel_atm_corr(test_heights)

    iag = np.array([])
    for hi in test_heights:
        corr = atm.iag_atm_corr_sph(atm.ussa76_density,
                                    hi, 84852, n_samples)
        iag = np.append(iag, corr)

    np.testing.assert_almost_equal(grs80, pz90, decimal=decimal)
    np.testing.assert_almost_equal(grs80, wenz, decimal=decimal)
    np.testing.assert_almost_equal(grs80, iag, decimal=decimal)
