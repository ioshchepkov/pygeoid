
import pytest
from pygeoid.constants.solar_system_gm import get_body_gm, gm_moon


def test_get_body_gm():
    with pytest.raises(ValueError):
        body = get_body_gm('no_name_body')

    body_gm = get_body_gm('moon')
    assert gm_moon == body_gm

