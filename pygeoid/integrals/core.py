"""Core classes for the geodetic integrals and their kernels.

"""

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle


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

    def _check_spherical_distance(self, spherical_distance):
        """Check spherical distance.

        """
        # if not np.logical_and(spherical_distance >= 0 * u.deg,
        #                      spherical_distance <= np.pi * u.rad).any():
        if not Angle(spherical_distance).is_within_bounds(0 * u.deg,
                                                          180 * u.deg):
            raise ValueError('spherical_distance must be between 0 and 180 degrees.')

        return spherical_distance


class Integral:
    """Base class for all integrals.

    """
    pass
