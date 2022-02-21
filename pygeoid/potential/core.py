
import abc
import uuid
from collections import OrderedDict
from functools import reduce

import astropy.units as u
import numpy as np

__all__ = ["PotentialBase", "CompositePotential"]


class PotentialBase(metaclass=abc.ABCMeta):
    """
    A baseclass for defining pure-Python gravitational potentials.

    The idea is heavily based on Gala: https://github.com/adrn/gala

    """

    _default_derivative_coordinates: str = "cartesian"
    """Default coordinate systen for calculating derivetives of the potential.

    """

    _default_gradient_coordinates: str = "cartesian"
    """Default coordinate system for calculating gradient vector components.

    """

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
        gradient_vector_method = '_gradient_vector_' + coordinates
        if gradient_vector_method in dir(self):
            grad = getattr(self, gradient_vector_method)(position=position,
                                                         *args, **kwargs)
        else:
            representation = position.represent_as(coordinates)
            deriv = {var: self.derivative(position, var,
                                          coordinates) for var in representation.components}
            scale_factors = representation.scale_factors()

            grad = {var: deriv[var] / scale_factors[var] for
                    var in representation.components}
        return grad

    def _hessian(self, position, *args, **kwargs):
        raise NotImplementedError

    @u.quantity_input
    def potential(self, position,
                  *args, **kwargs) -> u.m**2 / u.s**2:
        """Return potential value at given position.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame
            Position in ECEF or LocalTangentPlane frame.

        Returns
        -------
        potential : ~astropy.units.Quantity
            Potential value.

        """
        return self._potential(position=position, *args, **kwargs)

    def derivative(self, position, variable, coordinates=None, *args, **kwargs):
        """Return potential derivative.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame
            Position in ECEF or LocalTangentPlane frame.
        variable : str
            Name of the coordinate variable.
        corrdinates : str
            Name of the coordinate representation.

        Returns
        -------
        derivative : ~astropy.units.Quantity
            Potential derivative.

        """
        if coordinates is None:
            coordinates = self._default_derivative_coordinates

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
            coordinates = self._default_gradient_coordinates
        return self._gradient(position,
                              coordinates=coordinates, *args, **kwargs)

    def gradient_vector(self, position, coordinates=None, *args, **kwargs):
        """Return gradient vector in given coordinates.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in ECEF or LocalTangentPlane frame.
        corrdinates : str
            Name of the coordinate representation.

        """
        if coordinates is None:
            coordinates = self._default_gradient_coordinates
        return self._gradient_vector(position=position,
                                     coordinates=coordinates, *args, **kwargs)

    def hessian(self, position, *args, **kwargs):
        return self._hessian(position, *args, **kwargs)

    #########################################################################
    # Python's special methods
    #########################################################################
    def __call__(self, position):
        return self.potential(position)

    def __add__(self, other):
        if not isinstance(other, PotentialBase):
            raise TypeError(f'Cannot add a {self.__class__.__name__} to a '
                            f'{other.__class__.__name__}')

        new_pot = CompositePotential()

        if isinstance(self, CompositePotential):
            for k, v in self.items():
                new_pot[k] = v

        else:
            k = str(uuid.uuid4())
            new_pot[k] = self

        if isinstance(other, CompositePotential):
            for k, v in self.items():
                if k in new_pot:
                    raise KeyError(f'Potential component "{k}" already exists '
                                   '-- duplicate key provided in potential '
                                   'addition')
                new_pot[k] = v

        else:
            k = str(uuid.uuid4())
            new_pot[k] = other

        return new_pot

    def __str__(self):
        return self.__class__.__name__


class CompositePotential(PotentialBase, OrderedDict):
    """
    A potential composed of several distinct components. For example,
    two point masses or gravity potential (gravitational + centrifugal),
    each with their own potential model.

    The idea is heavily based on Gala: https://github.com/adrn/gala
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
