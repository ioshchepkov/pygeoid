import math

import pytest
from astropy import units as astropy_units

from pygeoid.conventions import units


def test_acceleration_aliases():
    assert units.m_per_s2 == astropy_units.m / astropy_units.s**2
    assert units.m2_per_s2 == astropy_units.m**2 / astropy_units.s**2


def test_eotvos_unit():
    assert units.E == units.Eotvos
    assert units.E.to(astropy_units.s**-2) == pytest.approx(1e-9)


def test_revolution_unit():
    assert units.rev == units.revolution
    assert units.rev.to(astropy_units.rad) == pytest.approx(math.tau)


def test_delegates_astropy_units():
    assert units.m is astropy_units.m
    assert units.Quantity is astropy_units.Quantity


def test_unknown_attribute_error():
    name = "not_a_real_unit"
    with pytest.raises(AttributeError, match=name):
        getattr(units, name)


def test_all_contains_pygeoid_units():
    assert set(units.__all__) == {
        "E",
        "Eotvos",
        "m_per_s2",
        "m2_per_s2",
        "rev",
        "revolution",
    }


def test_dir_contains_pygeoid_and_astropy_units():
    names = dir(units)

    assert "E" in names
    assert "Eotvos" in names
    assert "rev" in names
    assert "revolution" in names
    assert "m_per_s2" in names
    assert "m" in names
