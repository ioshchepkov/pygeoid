"""Units support for PyGeoid.

This module is a thin facade over :mod:`astropy.units`.
The Gal unit (1 Gal = 1 cm / s**2) is already there.

In addition, the Eotvos unit (1 E = 10**-9 s**-2) for the gravity
gradient is defined.

Use it as:

    from pygeoid.conventions import units as u

The objects are Astropy objects.
"""

import math as _math
from typing import Any

from astropy import units as _u

# TODO: re-export main objects and units from astropy.units and place them in __all__

# Define convenience aliases
m_per_s2 = _u.m / _u.s**2
m2_per_s2 = _u.m**2 / _u.s**2

# Define new units
E = Eotvos = _u.def_unit(
    ["E", "Eotvos"],
    represents=1e-9 * _u.s**-2,
    doc="Eötvös-unit for the gravity gradient.",
    prefixes=True,
)
rev = revolution = _u.def_unit(
    ["rev", "revolution"],
    represents=_math.tau * _u.rad,
    doc="Revolution unit for angles.",
)

__all__ = [
    "E",
    "Eotvos",
    "m_per_s2",
    "m2_per_s2",
    "rev",
    "revolution",
]


def __getattr__(name: str) -> Any:
    """Delegate unknown public names to astropy.units."""
    try:
        return getattr(_u, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    """Show both PyGeoid-defined and Astropy unit names."""
    return sorted(set(__all__) | set(dir(_u)))
