"""Core classes for the geodetic integrals and their kernels.

"""

import numpy as np


class Kernel:
    """Base class for all kernels.

    """

    _name = None

    @property
    def name(self):
        """Return kernel name.

        """
        return self._name


class SphericalKernel(Kernel):
    """Base class for all spherical kernels.

    """

    def _check_spherical_distance(self, spherical_distance, degrees):
        """Check spherical distance.

        """
        if degrees:
            spherical_distance = np.radians(spherical_distance)

        if not np.logical_and(spherical_distance >= 0,
                              spherical_distance <= np.pi).any():
            raise ValueError('spherical_distance must be between 0 and 180 degrees.')

        return spherical_distance


class Integral:
    """Base class for all integrals.

    """
    pass
