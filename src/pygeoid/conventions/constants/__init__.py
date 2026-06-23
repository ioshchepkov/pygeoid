"""Physical and geodetic constants.

This module delegates unknown names to :mod:`astropy.constants`.
"""

from typing import Any

from astropy import constants as _constants

__all__ = []


def __getattr__(name: str) -> Any:
    """Delegate unknown public names to astropy.constants."""
    try:
        return getattr(_constants, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    """Show Astropy constant names."""
    return sorted(set(__all__) | set(dir(_constants)))
