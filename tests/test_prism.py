
import astropy.units as u
import numpy as np
import pytest
from pygeoid.coordinates.frame import LocalFrame
from pygeoid.reductions import prism

bounds = u.Quantity(np.float64([0, 100, 0, 100, 0, 100]), u.m)
density = 2670 * u.kg / u.m**3
p = prism.Prism(bounds, density=density)

def test_bounds():

    with pytest.raises(ValueError):
        p.gxx(LocalFrame(50 * u.m, 50 * u.m, 50 * u.m))
    with pytest.raises(ValueError):
        p.gxx(LocalFrame(100 * u.m, 100 * u.m, 0 * u.m))


def test_gxx_gyy_gzz():

    nx = ny = nz = 100
    x = np.linspace(-1000, 1000, nx, dtype=np.float64)
    y = np.linspace(-1000, 1000, ny, dtype=np.float64)
    z = np.linspace(-1000, 1000, nz, dtype=np.float64)

    cond = (x <= 100) & (x >= 0)
    x = np.ma.masked_where(cond, x).compressed()
    y = np.ma.masked_where(cond, y).compressed()
    z = np.ma.masked_where(cond, z).compressed()

    xx, yy, zz = np.meshgrid(x, y, z) * u.m

    cart = LocalFrame(xx, yy, zz)

    gxx = p.gxx(cart).value
    gyy = p.gyy(cart).value
    gzz = p.gzz(cart).value

    # Laplace equation
    np.testing.assert_almost_equal(gxx + gyy, -gzz, decimal=10)
    np.testing.assert_almost_equal(gxx + gyy + gzz, 0, decimal=10)

    # Invariant 1 = Laplace equation
    i_1 = p.invariants(cart)[0].value
    np.testing.assert_almost_equal(gxx + gyy + gzz, i_1, decimal=10)
