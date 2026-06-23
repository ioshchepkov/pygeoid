import pytest
from astropy import constants as astropy_constants

from pygeoid.conventions import constants


def test_constant_aliases():
    assert constants.G is astropy_constants.G
    assert constants.g0 is astropy_constants.g0


def test_delegates_astropy_constants():
    assert constants.G is astropy_constants.G
    assert constants.g0 is astropy_constants.g0
    assert constants.c is astropy_constants.c
    assert constants.Constant is astropy_constants.Constant


def test_unknown_attribute_error():
    name = "not_a_real_constant"
    with pytest.raises(AttributeError, match=name):
        getattr(constants, name)


def test_all_is_empty():
    assert constants.__all__ == []


def test_dir_contains_astropy_constants():
    names = dir(constants)

    assert "G" in names
    assert "g0" in names
    assert "c" in names
