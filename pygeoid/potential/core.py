
import abc
import operator

from collections import OrderedDict
from functools import reduce

import numpy as np
import astropy.units as u


class PotentialBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _potential(self, position, *args, **kwargs):
        pass

    def _derivative(self, position, variable, coordinates, *args, **kwargs):
        method_name = '_derivative_' + coordinates
        if method_name not in dir(self):
            raise NotImplementedError('{0} derivatives are not '
                                      'implemented!'.format(coordinates))
        else:
            return getattr(self, method_name)(position=position,
                                              variable=variable, *args, **kwargs)

    def _gradient(self, position, coordinates, *args, **kwargs):
        vector = self._gradient_vector(position, coordinates=coordinates)
        return np.linalg.norm(u.Quantity(list(vector.values())), axis=0)

    def _gradient_vector(self, position, coordinates, *args, **kwargs):
        representation = position.represent_as(coordinates)
        deriv = {var: self.derivative(position, var,
                                      coordinates) for var in representation.components}
        scale_factors = representation.scale_factors()

        grad = {var: deriv[var] / scale_factors[var] for
                var in representation.components}

        return grad

    @u.quantity_input
    def potential(self, position,
                  *args, **kwargs) -> u.m**2 / u.s**2:
        """Return potential value.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        return self._potential(position=position, *args, **kwargs)

    def derivative(self, position, variable, coordinates=None, *args, **kwargs):
        if coordinates is None:
            coordinates = position.representation_type.get_name()

        return self._derivative(position=position, variable=variable,
                                coordinates=coordinates, *args, **kwargs)

    @u.quantity_input
    def gradient(self, position, coordinates=None,
                 *args, **kwargs) -> u.m / u.s**2:
        """Return gradient value.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF

        """
        if coordinates is None:
            coordinates = position.representation_type.get_name()
        return self._gradient(position,
                              coordinates=coordinates, *args, **kwargs)

    def gradient_vector(self, position, coordinates=None, *args, **kwargs):
        if coordinates is None:
            coordinates = position.representation_type.get_name()
        return self._gradient_vector(position=position,
                                     coordinates=coordinates, *args, **kwargs)

    def hessian(self, position, *args, **kwargs):
        raise NotImplementedError


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
        return u.Quantity([
            p._potential(position) for p in self.values()]).sum(axis=0)

    def _gradient_vector(self, position, coordinates=None):
        comp_vectors = [p.gradient_vector(position, coordinates)
                        for p in self.values()]
        dictf = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.items()),
                       comp_vectors)
        return dict(dictf)

    def _derivative(self, position, variable, coordinates=None):
        return u.Quantity([p._derivative(position, variable, coordinates)
                           for p in self.values()]).sum(axis=0)

    def __repr__(self):
        return "<CompositePotential {}>".format(",".join(self.keys()))
