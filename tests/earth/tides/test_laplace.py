from pygeoid.conventions.constants import de440
from pygeoid.earth.tides.laplace import LaplaceTidalEquation


def test_laplace_uses_de440_body_gm():
    tides = LaplaceTidalEquation()

    assert tides.bodies_gm == [de440.GM_moon, de440.GM_sun]
