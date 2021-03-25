# -*- coding: utf-8 -*-
"""Stokes integral and kernel.

"""

import numpy as np
import astropy.units as u

from pygeoid.integrals.core import SphericalKernel
from pygeoid.integrals.veningmeinesz import VeningMeineszKernel

__all__ = ['StokesKernel', 'StokesExtendedKernel']


class StokesKernel(SphericalKernel):
    r"""Stokes kernel class.

    """

    _name = 'Stokes'

    @u.quantity_input
    def kernel(self, spherical_distance: u.deg):
        r"""Evaluate Stokes spherical kernel.

        This method will calculate the original Stokes's function.

        Parameters
        ----------
        spherical_distance : ~astropy.units.Quantity
            Spherical distance, in radians.

        Notes
        -----
        In closed form, Stokes's kernel depends on the spherical distance
        :math:`\psi` by [1]_:

        .. math::
            S\left(\psi\right) = \dfrac{1}{\sin{(\psi / 2)}} - 6\sin{(\psi/2)}
            + 1 - 5\cos{\psi} - 3\cos{\psi} \ln{\left[\sin{(\psi/2)} +
            \sin^2{(\psi/2)}\right]}.

        References
        ----------
        .. [1] Heiskanen WA, Moritz H (1967) Physical geodesy.
            Freeman, San Francisco

        """
        psi = self._check_spherical_distance(
            spherical_distance=spherical_distance)

        spsi2 = np.sin(psi / 2)
        cpsi = np.cos(psi)

        return 1 + 1 / spsi2 - 6 * spsi2 - 5 * cpsi -\
            3 * cpsi * np.log(spsi2 + spsi2**2)

    @u.quantity_input
    def derivative_spherical_distance(self, spherical_distance):
        r"""Evaluate Stokes's spherical kernel derivative.

        The derivative of the Stokes function is the Vening-Meinesz function.

        Parameters
        ----------
        spherical_distance : ~astropy.units.Quantity
            Spherical distance.

        Notes
        -----
        The derivative of Stokes's kernel is the Vening-Meinesz and it  depends
        on the spherical distance :math:`\psi` by [1]_:

        .. math ::
            \dfrac{d S\left(\psi\right)}{d\psi} = - \dfrac{\cos{(\psi /
            2)}}{2\sin^2{(\psi / 2)}} + 8\sin{\psi} -
            6\cos{(\psi / 2)} - 3\dfrac{1 - \sin{(\psi / 2)}}{\sin{\psi}} +
            3\sin{\psi}\ln{\left[\sin{(\psi/2)} + \sin^2{(\psi/2)}\right]}.

        References
        ----------
        .. [1] Heiskanen WA, Moritz H (1967) Physical geodesy.
            Freeman, San Francisco

        """
        return VeningMeineszKernel().kernel(
            spherical_distance=spherical_distance)

    def _kernel_t(self, t):
        r"""Evaluate Stokes kernel for -1 <= t <= 1.

        Parameter `t` is usually chosen as the cosine of the spherical distance.

        Parameters
        ----------
        t : float or array_like of floats
            Input parameter.

        Notes
        -----
        Sometimes it is useful to replace the original Stokes
        function :math:`S\left( \psi \right)` of the spherical distance
        :math:`\psi` by the parametric Stokes function
        :math:`S\left( t \right)` of the paramater
        :math:`t = \cos{\psi}`, :math:`-1 \leq t \leq 1` [1]_:

        .. math ::
            S\left(t\right) = 1 - 5t - 3\sqrt{2\left(1-t\right)} +
            \sqrt{\dfrac{2}{1-t}} - 3t\ln{\dfrac{\sqrt{1-t}
            \left(\sqrt{2} + \sqrt{1-t}\right)}{2}}.

        References
        ----------
        .. [1] Hagiwara Y (1976) A new formula for evaluating the truncation
            error coefficient. Bulletin Géodésique 50:131–135.

        """
        if not np.logical_and(t >= -1, t <= 1).any():
            raise ValueError('t must be between -1 and 1.')

        sq2 = np.sqrt(2)
        sq1t = np.sqrt(1 - t)

        kernel = 1 - 5 * t - 3 * sq2 * sq1t + sq2 / sq1t
        kernel -= 3 * t * np.log(0.5 * (sq1t * (sq2 + sq1t)))

        return kernel

    def _derivative_t(self, t):
        r"""Stokes's function derivative with respect to -1 <= t <= 1.

        Parameter `t` is usually chosen as the cosine of the spherical distance.

        Parameters
        ----------
        t : float or array_like of floats
            Input parameter.

        Notes
        -----
        The derivative of the Stokes's kernel by
        :math:`t = \cos{\psi}`, :math:`-1 \leq t \leq 1` is [1]_:

        .. math ::
            \dfrac{d S\left(t\right)}{dt} = - 8 +
            \dfrac{3\sqrt{2}}{\sqrt{1-t}} +
            \dfrac{1}{\sqrt{2}\left(1-t\right)^{3/2}} +
            \dfrac{3\left(\sqrt{2} - \sqrt{1-t}\right)}
            {\sqrt{2}\left(1-t^2\right)} -
            3\ln{\dfrac{\sqrt{1-t}\left(\sqrt{2} + \sqrt{1-t}\right)}{2}}.

        References
        ----------
        .. [1] Hagiwara Y (1976) A new formula for evaluating the truncation
            error coefficient. Bulletin Géodésique 50:131–135.

        """
        if not np.logical_and(t >= -1, t <= 1).any():
            raise ValueError('t must be between -1 and 1.')

        sq2 = np.sqrt(2)
        sq1t = np.sqrt(1 - t)

        kernel = -8 + 3 * sq2 / sq1t + 1 / (sq2 * sq1t**3)
        kernel += 3 * (sq2 - sq1t) / (sq2 * (1 - t**2))
        kernel -= 3 * np.log(0.5 * (sq1t * (sq2 + sq1t)))

        return kernel


class StokesExtendedKernel(SphericalKernel):

    _name = 'Extended Stokes'

    def kernel(self, radius, spherical_distance):
        raise NotImplementedError

    def derivative_radius(self, radius, spherical_distance):
        raise NotImplementedError

    def derivative_spherical_distance(self, radius, spherical_distance):
        raise NotImplementedError
