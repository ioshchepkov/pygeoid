"""Base classes and utilities for coordinate containers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

__all__ = ["BaseCoordinates"]


@dataclass(frozen=True, slots=True)
class BaseCoordinates(ABC):
    """Base class for coordinate containers."""

    @classmethod
    @abstractmethod
    def from_cartesian(cls, coordinates) -> Self:
        """Create coordinates from Cartesian coordinates."""

    @abstractmethod
    def to_cartesian(self):
        """Convert these coordinates to Cartesian coordinates."""

    @abstractmethod
    def unit_vectors(self):
        """Return the local basis vectors as Cartesian components."""

    @abstractmethod
    def scale_factors(self):
        """Return the metric scale factors for the coordinate system."""
