
import os
import numpy as np
import pygeoid.reduction.topography as topo


def test_bouguer_plate():
    h_test = np.linspace(0, 10e3)
    gb7 = topo.bouguer_plate(h_test, 2670)
    gb3 = topo.bouguer_plate(h_test, 2300)
    gb7_from_gb3 = gb3 + 0.015515758 * h_test
    np.testing.assert_almost_equal(gb7, gb7_from_gb3, decimal=3)


def test_bullard_b():
    fname = os.path.join(os.path.dirname(__file__),
                         'data/Bullard_B_correction.txt')
    test_heights, test_bb = np.loadtxt(fname, unpack=True,
                                       delimiter='\t', skiprows=6, dtype=float)

    plate = topo.bouguer_plate(test_heights)
    cap = topo.spherical_bouguer_cap(test_heights)
    np.testing.assert_almost_equal(cap - plate, test_bb, decimal=2)
