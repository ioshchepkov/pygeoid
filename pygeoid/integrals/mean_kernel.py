"""Mean kernel computation for geodetic integrals.

References
----------
.. [1] Hirt C, Featherstone WE, Claessens SJ (2011) On the accurate numerical
    evaluation of geodetic convolution integrals. J Geod 85:519–538.
    https://doi.org/10.1007/s00190-011-0451-5

"""

import numpy as np
import numpy.ma as ma


from pygeoid.coordinates.transform import latlon_to_metres
from pygeoid.coordinates.utils import spherical_distance


class MeanKernel:
    r"""Base class for mean kernel computation.


    References
    ----------
    .. [1] Hirt C, Featherstone WE, Claessens SJ (2011) On the accurate numerical
        evaluation of geodetic convolution integrals. J Geod 85:519–538.
        https://doi.org/10.1007/s00190-011-0451-5

    """

    _asympotic_kernel_factor = None

    def asymptotic_kernel(self, spherical_distance, degrees=True, **kwargs):
        """Kernel in asymptotic approximation.

        """
        return self._asymptotic_kernel(spherical_distance,
                                       degrees=degrees, **kwargs)

    def planar_kernel(self, x, y, **kwargs):
        """Kernel in planar approximation.

        """
        return self._planar_kernel(x, y, **kwargs)

    def mean_planar_kernel(self, x1, y1, x2, y2):
        r"""Mean kernel in planar approximation.

        Parameters
        ----------
        x1, y1, x2, y2 : float or array_like of floats
            Boundaries coordinates of data point cell
            relative to computational point.

        Notes
        -----
        For a cell bounded by :math:`v = (x_1, y_1, x_2, y_2)`, the mean kernel
        :math:`\bar{K}_S (v)` in planar approximation is computed as [1]_:

        .. math::
            \bar{K}_S (v) = \dfrac{1}{\Delta x \Delta y} \left[
            F_s (x_1, y_1) - F_s (x_2, y_1) - F_s (x_1, y_2) + F_s (x_2, y_2)
            \right],

        where :math:`\Delta x\Delta y` is the surface area of the cell and
        :math:`F_s (x, y)` the antiderivative of the planar kernel.

        References
        ----------
        .. [1] Hirt C, Featherstone WE, Claessens SJ (2011) On the accurate numerical
            evaluation of geodetic convolution integrals. J Geod 85:519–538.
            https://doi.org/10.1007/s00190-011-0451-5

        """
        area = np.abs((x2 - x1) * (y2 - y1))
        fx1y1 = self._antiderivative_planar_kernel(x1, y1)
        fx2y1 = self._antiderivative_planar_kernel(x2, y1)
        fx1y2 = self._antiderivative_planar_kernel(x1, y2)
        fx2y2 = self._antiderivative_planar_kernel(x2, y2)
        return 1 / area * (fx1y1 - fx2y1 - fx1y2 + fx2y2)

    def _kernel_weighting_factor_planar(self, x, y, x1, y1, x2, y2):
        """Kernel weighting factor in planar approximation.

        The kernel weightening factor in planar approximation is the ratio of
        mean planar kernel of the cell to point planar kernel for the center
        of the cell.

        Parameters
        ----------
        x, y : float or array_like of floats
            Planar coordinates of the center of the data point cell
            relative to computational point.
        x1, y1, x2, y2 : float or array_like of floats
            Boundaries coordinates of data point cell
            relative to computational point.

        """
        planar_kernel = self._planar_kernel(x, y)
        mean_planar_kernel = self.mean_planar_kernel(x1, y1, x2, y2)
        return mean_planar_kernel / planar_kernel

    def mean_kernel(self, latp, lonp, latq, lonq, dlat, dlon):
        r"""Compute mean kernel for a rectangular cell.

        Parameters
        ----------
        latp, lonp : float or array_like of floats
            Latitudes and longitudes of the computational point(s),
            in degrees.
        latq, lonq : float or array_like of floats
            Latitude and longitude of the integration point(s), in degrees.
        dlat, dlon : float
            Cell size of the data grid, in degrees.

        References
        ----------
        .. [1] Hirt C, Featherstone WE, Claessens SJ (2011) On the accurate numerical
            evaluation of geodetic convolution integrals. J Geod 85:519–538.
            https://doi.org/10.1007/s00190-011-0451-5

        """

        # coordinates transformation
        origin = (latp, lonp)
        x, y = latlon_to_metres(latq, lonq, origin)

        # cell boundaries
        x1, y1 = latlon_to_metres(latq - 0.5 * dlat,
                lonq - 0.5 * dlon, origin)
        x2, y2 = latlon_to_metres(latq + 0.5 * dlat,
                lonq + 0.5 * dlon, origin)

        weight = self._kernel_weighting_factor_planar(x, y, x1, y1, x2, y2)

        # TODO: move to MeanSphericalKernel
        psi = spherical_distance(latp, lonp, latq, lonq,
                                 degrees=True).reshape(latq.shape)

        psi = ma.masked_outside(psi, self.psi1, self.psi2)

        kernel = self.kernel(psi, degrees=True)

        mean_kernel = weight * kernel

        return mean_kernel.filled(0.0)


class MeanSphericalKernel(MeanKernel):
    pass


class MeanStokesKernelBase(MeanKernel):
    def _asymptotic_kernel(self, spherical_distance, degrees=True):
        psi = self._check_spherical_distance(
            spherical_distance=spherical_distance,
            degrees=degrees)
        return self._asymptotic_kernel_factor / psi

    def _planar_kernel(self, x, y):
        dist = np.linalg.norm([x, y])
        masked = ma.masked_values(dist, 0.0)
        return self._asymptotic_kernel_factor / masked

    def _antiderivative_planar_kernel(self, x, y):
        dist = np.linalg.norm([x, y])
        xln = x * ma.log(y + dist)
        yln = y * ma.log(x + dist)
        return self._asymptotic_kernel_factor * (xln + yln)


class MeanStokesKernel(MeanStokesKernelBase):
    _asymptotic_kernel_factor = 2


class MeanHotineKernel(MeanStokesKernelBase):
    _asymptotic_kernel_factor = 2


class MeanEotvosKernel(MeanStokesKernelBase):
    _asymptotic_kernel_factor = 2


class MeanGreenMolodenskyKernel(MeanStokesKernelBase):
    _asymptotic_kernel_factor = 6


class MeanTidalDisplacementKernel(MeanStokesKernelBase):
    _asymptotic_kernel_factor = np.sqrt(2)


class MeanGravityOTLKernel(MeanStokesKernelBase):
    _asymptotic_kernel_factor = -0.5


class MeanVeningMeineszKernelBase(MeanKernel):
    def _asymptotic_kernel(self, spherical_distance, degrees=True):
        psi = self._check_spherical_distance(
            spherical_distance=spherical_distance,
            degrees=degrees)
        return self._asymptotic_kernel_factor / psi**2

    def _planar_kernel(self, x, y):
        dist = np.linalg.norm([x, y])
        masked = ma.masked_values(dist, 0.0)
        general_kernel = self.kernel_factor_c / masked**3
        xi = general_kernel * y
        eta = general_kernel * x
        return xi, eta

    def _antiderivative_planar_kernel(self, x, y):
        dist = np.linalg.norm([x, y])
        xi = -self._asymptotic_kernel_factor * ma.log(x + dist)
        eta = -self._asymptotic_kernel_factor * ma.log(y + dist)
        return xi, eta


class MeanVeningMeineszKernel(MeanVeningMeineszKernelBase):
    _asymptotic_kernel_factor = -2


class MeanInverseVeningMeineszKernel(MeanVeningMeineszKernelBase):
    _asymptotic_kernel_factor = -2


class MeanTerrainEffectKernelBase(MeanKernel):
    def _asymptotic_kernel(self, distance, degrees=True):
        masked = ma.masked_values(distance, 0.0)
        return self._asymptotic_kernel_factor / masked**3

    def _planar_kernel(self, x, y):
        dist = np.linalg.norm([x, y])
        masked = ma.masked_values(dist, 0.0)
        return self.kernel_factor_c / masked**3

    def _antiderivative_planar_kernel(self, x, y):
        dist = np.linalg.norm([x, y])
        return -self._asymptotic_kernel_factor * dist / (x * y)


class MeanTerrainCorrectionKernel(MeanTerrainEffectKernelBase):
    _asymptotic_kernel_factor = 1


class MeanIndirectEffectKernel(MeanTerrainEffectKernelBase):
    _asymptotic_kernel_factor = 1


class MeanMolodenskyG1Kernel(MeanTerrainEffectKernelBase):
    _asymptotic_kernel_factor = 1
