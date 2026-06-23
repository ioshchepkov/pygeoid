"""Core classes for the geodetic integrals and their kernels."""

from astropy.coordinates import Angle

from pygeoid.conventions import units as u


class Kernel:
    """Base class for all kernels."""

    _name = None

    @property
    def name(self):
        """Return kernel name."""
        return self._name


class SphericalKernel(Kernel):
    """Base class for all spherical kernels."""

    def _check_spherical_distance(self, spherical_distance, *args, **kwargs):
        """Check spherical distance."""
        # if not np.logical_and(spherical_distance >= 0 * u.deg,
        #                      spherical_distance <= np.pi * u.rad).any():
        if not Angle(spherical_distance).is_within_bounds(0 * u.deg, 180 * u.deg):
            raise ValueError("spherical_distance must be between 0 and 180 degrees.")

        return spherical_distance

    def plot_kernel(self, ax=None):
        raise NotImplementedError


class Integral(Kernel):
    """Base class for all integrals."""

    pass


class SphericalIntegral(SphericalKernel):
    pass
