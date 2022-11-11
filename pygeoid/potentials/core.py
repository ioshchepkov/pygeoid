
import abc
import astropy.units as u
import numpy as np

from astropy.coordinates import BaseDifferential
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import CartesianDifferential

__all__ = ["PotentialBase"]


class PotentialBase(metaclass=abc.ABCMeta):
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

    @u.quantity_input
    def potential(self, position,
                  *args, **kwargs) -> u.m**2 / u.s**2:
        """Return potential value at given position.

        Parameters
        ----------
        position : subclass of BaseCoordinateFrame or BaseRepresentation
            Coordinate frame instance.

        Returns
        -------
        potential : ~astropy.units.Quantity
            Potential value.

        """
        return self._potential(position=position, *args, **kwargs)

    def differential(self, position, differential_class=None, **kwargs):
        """Return potential differential for a given representation.

        Parameters
        ----------
        position : subclass of BaseCoordinateFrame or BaseRepresentation
            Coordinate frame instance.
        differential_class : subclass of `~astropy.coordinates.BaseDifferential`, optional
            Class in which the differentials should be represented.

        """
        default_differential = self._differential(position, **kwargs)

        if differential_class is not None:
            if isinstance(differential_class, BaseDifferential):
                raise ValueError("""`differential_class` must be a subclass of
                        BaseDifferential.""")
            return default_differential.represent_as(
                differential_class,
                base=position.represent_as(
                    default_differential.base_representation))
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

        cart_diff = differential.represent_as(CartesianDifferential,
                                              base=position.represent_as(
                                                  differential.base_representation))
        return cart_diff.norm()

    def hessian(self, position, *args, **kwargs):
        """Return Hessian.

        Hessian is an Eotvos tensor.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        return self._hessian(position, *args, **kwargs)

    #########################################################################
    # Python's special methods
    #########################################################################
    def __str__(self):
        return self.__class__.__name__
