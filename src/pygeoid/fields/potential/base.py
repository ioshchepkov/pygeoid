"""Potential field base classes."""

import abc

from astropy.coordinates import (
    BaseDifferential,
    CartesianDifferential,
    CartesianRepresentation,
    PhysicsSphericalRepresentation,
    SphericalRepresentation,
)

from pygeoid.conventions import units as u
from pygeoid.fields.core import ScalarField

__all__ = ["PotentialBase"]


class PotentialBase(ScalarField, metaclass=abc.ABCMeta):
    """
    A baseclass for defining gravitational potentials.

    """

    @abc.abstractmethod
    def _potential(self, position, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _differential(self, position, *args, **kwargs):
        pass

    def _hessian(self, position, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _base_representation(position, representation):
        """Build the Astropy base required to transform a differential."""
        if not hasattr(position, "coordinates"):
            return position.represent_as(representation)

        if issubclass(representation, CartesianRepresentation):
            cartesian = position.cartesian
            return CartesianRepresentation(cartesian.x, cartesian.y, cartesian.z)
        if issubclass(representation, PhysicsSphericalRepresentation):
            spherical = position.spherical
            return PhysicsSphericalRepresentation(
                spherical.lon, 90 * u.deg - spherical.lat, spherical.radius
            )
        if issubclass(representation, SphericalRepresentation):
            spherical = position.spherical
            return SphericalRepresentation(
                spherical.lon, spherical.lat, spherical.radius
            )
        raise TypeError(f"Unsupported differential base: {representation!r}")

    @u.quantity_input
    def potential(self, position, *args, **kwargs) -> u.m**2 / u.s**2:
        """Return potential value at given position.

        Parameters
        ----------
        position : subclass of BaseCoordinateFrame or BaseRepresentation
            Coordinate frame instance.

        Returns
        -------
        potential : ~pygeoid.conventions.units.Quantity
            Potential value.

        """
        return self._potential(position=position, *args, **kwargs)

    def value(self, position, *args, **kwargs):
        """Return potential value at given position."""
        return self.potential(position, *args, **kwargs)

    def differential(self, position, differential_class=None, **kwargs):
        """Return potential differential for a given representation.

        Parameters
        ----------
        position : subclass of BaseCoordinateFrame or BaseRepresentation
            Coordinate frame instance.
        differential_class : subclass of `~astropy.coordinates.BaseDifferential`,
        optional
            Class in which the differentials should be represented.

        """
        default_differential = self._differential(position, **kwargs)

        if differential_class is not None:
            if isinstance(differential_class, BaseDifferential):
                raise ValueError("""`differential_class` must be a subclass of
                        BaseDifferential.""")
            return default_differential.represent_as(
                differential_class,
                base=self._base_representation(
                    position, default_differential.base_representation
                ),
            )
        else:
            return default_differential

    @u.quantity_input
    def gradient(self, position, *args, **kwargs) -> u.m / u.s**2:
        """Return gradient value.

        Parameters
        ----------
        position : subclass of BaseCoordinateFrame or BaseRepresentation
            Coordinate frame instance.

        """
        differential = self.differential(position, *args, **kwargs)

        cart_diff = differential.represent_as(
            CartesianDifferential,
            base=self._base_representation(position, differential.base_representation),
        )
        return cart_diff.norm()

    def hessian(self, position, *args, **kwargs):
        """Return Hessian.

        Hessian is an Eotvos tensor.

        Parameters
        ----------
        position : ~pygeoid.geometry.Position

        """
        return self._hessian(position, *args, **kwargs)
