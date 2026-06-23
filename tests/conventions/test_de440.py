from pygeoid.conventions import units as u
from pygeoid.conventions.constants import de440


def test_de440_constants_are_exported():
    assert "GM_sun" in de440.__all__
    assert de440.GM_sun.abbrev == "GM_sun"
    assert de440.GM_sun.value == 132712440041.279419
    assert de440.GM_sun.unit == u.km**3 / u.s**2
    assert de440.GM_sun.uncertainty == ""
    assert de440.GM_sun.system == "si"


def test_de440_table_reference_is_default_reference():
    assert "Table 2" not in de440.DE440.default_reference
    assert de440.GM_jupiter_system.reference == de440.DE440.default_reference


def test_de440_constants_keep_value_source_references():
    assert de440.GM_sun.reference.startswith(de440.DE440.default_reference)
    assert "Value source: estimated from DE440." in de440.GM_sun.reference
