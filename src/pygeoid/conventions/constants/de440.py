"""DE440 and DE441 solar-system gravitational parameters."""

from astropy.constants import Constant as _Constant


class DE440(_Constant):
    default_reference = (
        "Park, R. S., Folkner, W. M., Williams, J. G., & Boggs, D. H. "
        "2021, The JPL Planetary and Lunar Ephemerides DE440 and DE441, "
        "The Astronomical Journal, 161, 105, "
        "doi:10.3847/1538-3881/abd414. "
    )
    _registry = {}
    _has_incompatible_units = set()


GM_sun = DE440(
    abbrev="GM_sun",
    name="Solar gravitational mass parameter used in DE440 and DE441",
    value=132712440041.279419,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference + "Value source: estimated from DE440.",
    system="si",
)

GM_mercury = DE440(
    abbrev="GM_mercury",
    name="Mercury gravitational mass parameter used in DE440 and DE441",
    value=22031.868551,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value source: Konopliv, A. S., Park, R. S., & Ermakov, A. I. "
    "2020, Icarus, 335, 113386.",
    system="si",
)

GM_venus = DE440(
    abbrev="GM_venus",
    name="Venus gravitational mass parameter used in DE440 and DE441",
    value=324858.592000,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value source: Konopliv, A. S., Banerdt, W. B., & Sjogren, W. L. "
    "1999, Icarus, 139, 3.",
    system="si",
)

GM_earth = DE440(
    abbrev="GM_earth",
    name="Earth gravitational mass parameter used in DE440 and DE441",
    value=398600.435507,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference + "Value source: estimated from DE440.",
    system="si",
)

GM_mars_system = DE440(
    abbrev="GM_mars_system",
    name="Mars system gravitational mass parameter used in DE440 and DE441",
    value=42828.375816,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value source: Konopliv, A. S., Park, R. S., & Folkner, W. M. "
    "2016, Icarus, 274, 253.",
    system="si",
)

GM_jupiter_system = DE440(
    abbrev="GM_jupiter_system",
    name="Jupiter system gravitational mass parameter used in DE440 and DE441",
    value=126712764.100000,
    unit="km3 / s2",
    uncertainty="",
    system="si",
)

GM_saturn_system = DE440(
    abbrev="GM_saturn_system",
    name="Saturn system gravitational mass parameter used in DE440 and DE441",
    value=37940584.841800,
    unit="km3 / s2",
    uncertainty="",
    system="si",
)

GM_uranus_system = DE440(
    abbrev="GM_uranus_system",
    name="Uranus system gravitational mass parameter used in DE440 and DE441",
    value=5794556.400000,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value source: Jacobson, R. A. 2014, The Astronomical Journal, "
    "148, 76.",
    system="si",
)

GM_neptune_system = DE440(
    abbrev="GM_neptune_system",
    name="Neptune system gravitational mass parameter used in DE440 and DE441",
    value=6836527.100580,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value source: Jacobson, R. A. 2009, The Astronomical Journal, "
    "137, 4322.",
    system="si",
)

GM_pluto_system = DE440(
    abbrev="GM_pluto_system",
    name="Pluto system gravitational mass parameter used in DE440 and DE441",
    value=975.500000,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value source: Brozovic, M., Showalter, M. R., Jacobson, R. A., "
    "& Buie, M. W. 2015, Icarus, 246, 317.",
    system="si",
)

GM_moon = DE440(
    abbrev="GM_moon",
    name="Moon gravitational mass parameter used in DE440 and DE441",
    value=4902.800118,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference + "Value source: estimated from DE440.",
    system="si",
)

GM_ceres = DE440(
    abbrev="GM_ceres",
    name="Ceres gravitational mass parameter used in DE440 and DE441",
    value=62.62890,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value sources: Park, R. S., Konopliv, A. S., Bills, B. G., "
    "et al. 2016, Nature, 537, 515. "
    "Konopliv, A. S., Park, R. S., Vaughan, A. T., et al. "
    "2018, Icarus, 299, 411. "
    "Park, R. S., Vaughan, A. T., Konopliv, A. S., et al. "
    "2019, Icarus, 319, 812. "
    "Park, R. S., Konopliv, A. S., Ermakov, A. I., et al. "
    "2020a, Nature Astronomy, 4, 748.",
    system="si",
)

GM_vesta = DE440(
    abbrev="GM_vesta",
    name="Vesta gravitational mass parameter used in DE440 and DE441",
    value=17.288245,
    unit="km3 / s2",
    uncertainty="",
    reference=DE440.default_reference
    + "Value sources: Konopliv, A. S., Asmar, S. W., Park, R. S., "
    "et al. 2014, Icarus, 240, 103. "
    "Park, R. S., Konopliv, A. S., Asmar, S. W., et al. "
    "2014, Icarus, 240, 118.",
    system="si",
)


__all__ = (
    "GM_sun",
    "GM_mercury",
    "GM_venus",
    "GM_earth",
    "GM_mars_system",
    "GM_jupiter_system",
    "GM_saturn_system",
    "GM_uranus_system",
    "GM_neptune_system",
    "GM_pluto_system",
    "GM_moon",
    "GM_ceres",
    "GM_vesta",
)
