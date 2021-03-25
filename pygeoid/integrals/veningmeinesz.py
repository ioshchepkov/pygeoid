# -*- coding: utf-8 -*-
"""Vening Meinesz kernel.

"""

import numpy as np
import astropy.units as u

from pygeoid.integrals.core import SphericalKernel

__all__ = ['VeningMeineszKernel']


class VeningMeineszKernel(SphericalKernel):
    r"""Vening Meinesz kernel class.

    """

    _name = 'Vening Meinesz'

    @u.quantity_input
    def kernel(self, spherical_distance: u.deg):
        r"""Evaluate Vening Meinesz kernel.

        Parameters
        ----------
        spherical_distance : float or array_like of floats
            Spherical distance, in radians.
        degrees : bool, optional
            If True, the spherical distance is given in degrees,
            otherwise radians.

        Notes
        -----
        The derivative  is the Vening-Meinesz and it  depends
        on the spherical distance :math:`\psi` by [1]_:

        .. math ::
            V\left(\psi\right) =
            \dfrac{d S\left(\psi\right)}{d\psi} = - \dfrac{\cos{(\psi /
            2)}}{2\sin^2{(\psi / 2)}} + 8\sin{\psi} -
            6\cos{(\psi / 2)} - 3\dfrac{1 - \sin{(\psi / 2)}}{\sin{\psi}} +
            3\sin{\psi}\ln{\left[\sin{(\psi/2)} + \sin^2{(\psi/2)}\right]},

        which is the derivative of Stokes's kernel with respect to
        :math:`\psi`.

        References
        ----------
        .. [1] Heiskanen WA, Moritz H (1967) Physical geodesy.
            Freeman, San Francisco

        """

        psi = self._check_spherical_distance(
            spherical_distance=spherical_distance)

        cpsi2 = np.cos(psi / 2)
        spsi2 = np.sin(psi / 2)
        spsi = np.sin(psi)

        return -0.5 * cpsi2 / spsi2**2 + 8 * spsi - 6 * cpsi2 -\
            3 * (1 - spsi2) / spsi + 3 * spsi * np.log(spsi2 + spsi2**2)
