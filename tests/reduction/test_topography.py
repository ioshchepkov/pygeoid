
import os
import numpy as np
import astropy.units as u
import pygeoid.reduction.topography as topo


def test_bouguer_plate():
    h_test = np.linspace(0, 10e3) * u.m
    gb7 = topo.bouguer_plate(h_test, 2670 * u.kg / u.m**3)
    gb3 = topo.bouguer_plate(h_test, 2300 * u.kg / u.m**3)
    gb7_from_gb3 = (gb3.to('mGal').value + 0.015515758 * h_test.value) * u.mGal
    np.testing.assert_almost_equal(gb7.to('mGal').value, gb7_from_gb3.to('mGal').value, decimal=2)


def test_bullard_b():
    fname = os.path.join(os.path.dirname(__file__),
                         'data/Bullard_B_correction.txt')
    test_heights, test_bb = np.loadtxt(fname, unpack=True,
                                       delimiter='\t', skiprows=6, dtype=float)

    test_heights *= u.m
    test_bb *= u.mGal

    plate = topo.bouguer_plate(test_heights)
    cap = topo.spherical_bouguer_cap(test_heights)
    np.testing.assert_almost_equal((cap - plate).to('mGal').value,
            test_bb.to('mGal').value, decimal=2)
