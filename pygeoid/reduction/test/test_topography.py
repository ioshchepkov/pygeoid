
import numpy as np
import pygeoid.reduction.topography as topo


def test_bouguer_plate():
    h_test = np.linspace(0, 10e3)
    gb7 = topo.bouguer_plate(h_test, 2670)
    gb3 = topo.bouguer_plate(h_test, 2300)
    gb7_from_gb3 = gb3 + 0.015515758*h_test
    np.testing.assert_almost_equal(gb7, gb7_from_gb3, decimal=3)
