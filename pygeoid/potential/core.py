
import abc

import numpy as np
import astropy.units as u


class PotentialBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _potential(self, position, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _gradient(self, position, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _gradient_vector(self, position, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _derivative(self, position, *args, **kwargs):
        pass

    def potential(self, position, *args, **kwargs):
        """Return potential value.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        return self._potential(position=position, *args, **kwargs)

    def gradient(self, position, *args, **kwargs):
        """Return gradient value.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        return self._gradient(position=position, *args, **kwargs)

    def gradient_vector(self, position, coordinates=None, *args, **kwargs):
        if coordinates is None:
            coordinates = position.representation_type.get_name()

        return self._gradient_vector(position=position,
                                     coordinates=coordinates, *args, **kwargs)

    def derivative(self, position, variable, coordinates=None, *args, **kwargs):
        if coordinates is None:
            coordinates = position.representation_type.get_name()

        return self._derivative(position=position, variable=variable,
                                coordinates=coordinates, *args, **kwargs)
