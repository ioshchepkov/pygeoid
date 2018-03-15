
import pytest
import numpy as np
from pygeoid.simple import prism


def test_bounds():
    bounds = (0, 100, 0, 100, 0, 100)
    density = 2670
    p = prism.Prism(bounds, density=density)
    with pytest.raises(ValueError):
        p.gxx(50, 50, 50)
    with pytest.raises(ValueError):
        p.gxx(100, 100, 0)


def test_gxx_gyy_gzz():

    bounds = (0, 100, 0, 100, 0, 100)
    density = 2670
    p = prism.Prism(bounds, density=density)

    nx = ny = nz = 100
    x = np.linspace(-1000, 1000, nx, dtype=np.float64)
    y = np.linspace(-1000, 1000, ny, dtype=np.float64)
    z = np.linspace(-1000, 1000, nz, dtype=np.float64)

    cond = (x <= 100) & (x >= 0)
    x = np.ma.masked_where(cond, x).compressed()
    y = np.ma.masked_where(cond, y).compressed()
    z = np.ma.masked_where(cond, z).compressed()

    xx, yy, zz = np.meshgrid(x, y, z)

    gxx = p.gxx(xx, yy, zz).magnitude
    gyy = p.gyy(xx, yy, zz).magnitude
    gzz = p.gzz(xx, yy, zz).magnitude

    np.testing.assert_almost_equal(gxx + gyy, -gzz, decimal=10)
