
import pytest
from pygeoid.constants.solar_system_gm import get_body_gm


def test_get_body():
    with pytest.raises(ValueError):
        body = get_body_gm('no_name_body')

