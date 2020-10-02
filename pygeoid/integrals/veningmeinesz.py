# -*- coding: utf-8 -*-
"""Vening Meinesz kernel.

"""

import numpy as np

from pygeoid.integrals.core import SphericalIntegral
from pygeoid.integrals.mean_kernel import MeanVeningMeineszKernel
from pygeoid.integrals.mean_kernel import MeanInverseVeningMeineszKernel
from pygeoid.coordinates.utils import check_spherical_distance

__all__ = ['VeningMeinesz', 'InverseVeningMeinesz']


class VeningMeinesz(SphericalIntegral, MeanVeningMeineszKernel):
    r"""Vening Meinesz kernel class.

    """

    _name = 'Vening Meinesz'

    @staticmethod
    def kernel(spherical_distance, azimuth=None, degrees=True):
        r"""Evaluate Vening Meinesz kernel.

        Parameters
        ----------
        spherical_distance : float or array_like of floats
            Spherical distance, in radians.
        azimuth : float or array_like of floats, optional
            The azimuth from computation point to the data point.
            If defined, then both kernels for vertical deflections (xi, eta)
            are returned. Otherwise, only Vening Meinesz function is returned.
        degrees : bool, optional
            If True, the spherical distance and the azimuth are
            given in degrees, otherwise radians.

        Returns
        -------
        kernels : float or array_like of floats
            Vening Meinesz function or kernel values for vertical deflections
            (xi, eta) if azimuth is not None.

        Notes
        -----
        The Vening Meinesz kernel depends
        on the spherical distance :math:`\psi` by [1]_:

        .. math ::

            V\left(\psi\right) =
            \dfrac{d S\left(\psi\right)}{d\psi} = &- \dfrac{\cos{(\psi /
            2)}}{2\sin^2{(\psi / 2)}} + 8\sin{\psi} -
            6\cos{(\psi / 2)} \\
                    &- 3\dfrac{1 - \sin{(\psi / 2)}}{\sin{\psi}} +
            3\sin{\psi}\ln{\left[\sin{(\psi/2)} + \sin^2{(\psi/2)}\right]},

        which is the derivative of Stokes's kernel with respect to
        :math:`\psi`.

        If azimuth :math:`\alpha` is defined, then kernels for vertical
        deflections :math:`(\xi, \eta)` are
        :math:`V\left(\psi\right)\cos{\alpha}` and
        :math:`V\left(\psi\right)\sin{\alpha}`.

        References
        ----------
        .. [1] Heiskanen WA, Moritz H (1967) Physical geodesy.
            Freeman, San Francisco

        """

        psi = check_spherical_distance(
            spherical_distance=spherical_distance,
            degrees=degrees)

        cpsi2 = np.cos(psi / 2)
        spsi2 = np.sin(psi / 2)
        spsi = np.sin(psi)

        kernel = -0.5 * cpsi2 / spsi2**2 + 8 * spsi - 6 * cpsi2 -\
            3 * (1 - spsi2) / spsi + 3 * spsi * np.log(spsi2 + spsi2**2)

        if azimuth is not None:
            kernel_xi = kernel * np.cos(azimuth)
            kernel_eta = kernel * np.sin(azimuth)
            return kernel_xi, kernel_eta
        else:
            return kernel


class InverseVeningMeinesz(SphericalIntegral, MeanInverseVeningMeineszKernel):
    r"""Inverse Vening Meinesz kernel class.

    """

    _name = 'Inverse Vening Meinesz'

    def kernel(self, spherical_distance, degrees=True):
        r"""Evaluate Inverse Vening Meinesz kernel.

        Parameters
        ----------
        spherical_distance : float or array_like of floats
            Spherical distance, in radians.
        degrees : bool, optional
            If True, the spherical distance is given in degrees,
            otherwise radians.

        Notes
        -----
        The Inverse Vening-Meinesz kernel depends
        on the spherical distance :math:`\psi` by [1]_:

        .. math ::

            H'\left(\psi\right) =
            -\dfrac{\cos{(\psi/2)}}{2\sin^2{(\psi/2)}} +
            \dfrac{\cos{(\psi/2)}\left[3 +
            2\sin{(\psi/2)}\right]}{2\sin{(\psi/2)}\left[1 + \sin{(\psi/2)}\right]}.

        References
        ----------
        .. [1] Hirt C, Featherstone WE, Claessens SJ (2011) On the accurate numerical
            evaluation of geodetic convolution integrals. J Geod 85:519–538.
            https://doi.org/10.1007/s00190-011-0451-5

        """

        psi = self._check_spherical_distance(
            spherical_distance=spherical_distance,
            degrees=degrees)

        cpsi2 = np.cos(psi / 2)
        spsi2 = np.sin(psi / 2)

        return -cpsi2 / (2 * spsi2**2) + cpsi2 * (3 +
                                                  2 * spsi2) / (2 * spsi2 * (1 + spsi2))
