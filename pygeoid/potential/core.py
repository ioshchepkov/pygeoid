
import abc

from collections import OrderedDict

import numpy as np
import astropy.units as u


class PotentialBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _potential(self, position, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _gradient(self, position, *args, **kwargs):
        pass

    #@abc.abstractmethod
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
        return self._gradient(position, *args, **kwargs)

    def gradient_vector(self, position, coordinates=None, *args, **kwargs):
        if coordinates is None:
            coordinates = position.representation_type.get_name()

        position = position.represent_as(coordinates)
        deriv = {var:self._derivative(position, var,
            coordintaes) for var in position.components}
        scale_factors = position.scale_factors()

        grad = {variable:deriv[var]/scale_factors[var] for
                var in position.components}

        return grad

    def derivative(self, position, variable, coordinates=None, *args, **kwargs):
        if coordinates is None:
            coordinates = position.representation_type.get_name()

        return self._derivative(position=position.represent_as(coordinates),
                variable=variable, coordinates=coordinates, *args, **kwargs)


class CompositePotential(PotentialBase, OrderedDict):
    """
    A potential composed of several distinct components. For example,
    two point masses or gravity potential (gravitational + centrifugal),
    each with their own potential model.

    """

    def __init__(self, *args, **kwargs):

        if len(args) > 0 and isinstance(args[0], list):
            for k, v in args[0]:
                kwargs[k] = v
        else:
            for i, v in args:
                kwargs[str(i)] = v

        for v in kwargs.values():
            self._check_component(v)

        OrderedDict.__init__(self, **kwargs)

    def __setitem__(self, key, value):
        self._check_component(value)
        super(CompositePotential, self).__setitem__(key, value)

    def _check_component(self, p):
        if not isinstance(p, PotentialBase):
            raise TypeError("Potential components may only be Potential "
                            "objects, not {0}.".format(type(p)))

    def _potential(self, position):
        return np.sum([p._potential(position) for p in self.values()], axis=0)

    def _gradient(self, position, coordinates=None):
        vector = self._gradient_vector(position, coordinates=coordinates)
        return np.linalg.norm(vector, axis=0)

    def _gradient_vector(self, position, coordinates=None):
        return u.Quantity([u.Quantity(p._gradient_vector(position,
                                                         coordinates)) for p in self.values()]).sum(axis=0)

    def _derivative(self, position, variable, coordinates=None):
        return np.sum([p._derivative(position, variable, coordinates)
                       for p in self.values()], axis=0)

    def __repr__(self):
        return "<CompositePotential {}>".format(",".join(self.keys()))
