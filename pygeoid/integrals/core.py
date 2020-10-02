"""Core classes for the geodetic integrals.

"""

import numpy as np


class Integral:
    """Base class for all kernels.

    """

    _name = None

    @property
    def name(self):
        """Return kernel name.

        """
        return self._name


class SphericalKernel:
    def plot_kernel(self, ax=None):
        raise NotImplementedError


class SphericalIntegral(Integral):
    """Base class for all spherical integrals.

    """
    pass
