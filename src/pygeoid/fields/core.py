"""Base classes for scalar, vector, and tensor fields."""

import abc

__all__ = ["FieldBase", "ScalarField", "VectorField", "TensorField"]


class FieldBase(metaclass=abc.ABCMeta):
    """Base class for fields."""

    def __str__(self):
        return self.__class__.__name__


class ScalarField(FieldBase):
    """Base class for scalar fields."""

    @abc.abstractmethod
    def value(self, position, *args, **kwargs):
        """Return scalar field value at a given position."""
        pass


class VectorField(FieldBase):
    """Base class for vector fields."""

    @abc.abstractmethod
    def vector(self, position, *args, **kwargs):
        """Return vector field value at a given position."""
        pass


class TensorField(FieldBase):
    """Base class for tensor fields."""

    @abc.abstractmethod
    def tensor(self, position, *args, **kwargs):
        """Return tensor field value at a given position."""
        pass
