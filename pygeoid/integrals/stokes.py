# -*- coding: utf-8 -*-
"""Stokes integral.

"""

import numpy as np
import xarray as xr

from scipy.interpolate import bisplrep, bisplev
from tqdm import tqdm

from pygeoid.integrals.core import SphericalIntegral
from pygeoid.integrals.mean_kernel import MeanStokesKernel
from pygeoid.integrals.veningmeinesz import VeningMeinesz
from pygeoid.coordinates.utils import (spherical_distance,
        check_spherical_distance)
from pygeoid.constants import R_earth, _4pi


__all__ = ['Stokes']


class Stokes(SphericalIntegral, MeanStokesKernel):
    r"""Stokes (spherical) integral.

    """

    _name = 'Stokes'

    def __init__(self, data, psi1=0.0, psi2=180.0,
            reference_radius=6371e3):

        if not isinstance(data, xr.DataArray):
            raise TypeError("data must be instance of xarray.DataArray")

        self.data = data
        self.psi1 = psi1
        self.psi2 = psi2
        self.ref_radius = reference_radius
        self._pre_factor = self.ref_radius / _4pi

    @staticmethod
    def kernel(spherical_distance, degrees=True):
        r"""Evaluate Stokes spherical kernel.

        This method will calculate the original Stokes's function.

        Parameters
        ----------
        spherical_distance : float or array_like of floats
            Spherical distance, in radians.
        degrees : bool, optional
            If True, the spherical distance is given in degrees,
            otherwise radians.

        Notes
        -----
        In closed form, Stokes's kernel depends on the spherical distance
        :math:`\psi` by [1]_:

        .. math::

            S\left(\psi\right) = & \dfrac{1}{\sin{(\psi / 2)}} - 6\sin{(\psi/2)}
            + 1 - 5\cos{\psi} \\
                    &- 3\cos{\psi} \ln{\left[\sin{(\psi/2)} +
            \sin^2{(\psi/2)}\right]}.

        References
        ----------
        .. [1] Heiskanen WA, Moritz H (1967) Physical geodesy.
            Freeman, San Francisco

        """
        psi = check_spherical_distance(
                spherical_distance=spherical_distance,
                degrees=degrees)

        spsi2 = np.sin(psi / 2)
        cpsi = np.cos(psi)

        return 1 + 1 / spsi2 - 6 * spsi2 - 5 * cpsi -\
                3 * cpsi * np.log(spsi2 + spsi2**2)

    @staticmethod
    def derivative_spherical_distance(spherical_distance, degrees=True):
        r"""Evaluate Stokes's spherical kernel derivative.

        The derivative of the Stokes function is the Vening-Meinesz function.

        Parameters
        ----------
        spherical_distance : float or array_like of floats
            Spherical distance.
        degrees : bool, optional
            If True, the input `psi` is given in degrees, otherwise radians.

        Notes
        -----
        The derivative of Stokes's kernel is the Vening-Meinesz and it  depends
        on the spherical distance :math:`\psi` by [1]_:

        .. math ::

            \dfrac{d S\left(\psi\right)}{d\psi} = &- \dfrac{\cos{(\psi /
            2)}}{2\sin^2{(\psi / 2)}} + 8\sin{\psi} -
            6\cos{(\psi / 2)} \\
                    &- 3\dfrac{1 - \sin{(\psi / 2)}}{\sin{\psi}} +
            3\sin{\psi}\ln{\left[\sin{(\psi/2)} + \sin^2{(\psi/2)}\right]}.

        References
        ----------
        .. [1] Heiskanen WA, Moritz H (1967) Physical geodesy.
            Freeman, San Francisco

        """
        return VeningMeinesz.kernel(
                spherical_distance=spherical_distance,
                degrees=degrees)

    @staticmethod
    def _kernel_t(t):
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

    @staticmethod
    def _derivative_t(t):
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

    def innermost_interpolation(self, latP, lonP, nzn, dlat, dlon):
        # innermost zone
        lat_iz_bnds = [latP + dlat * nzn, latP - dlat * nzn]
        lon_iz_bnds = [lonP - dlon * nzn, lonP + dlon * nzn]
        iz_grid = self.data.rgrid.sel(lat=slice(*lat_iz_bnds),
                lon=slice(*lon_iz_bnds))

        data = iz_grid[::-1].data
        lons, lats = np.meshgrid(iz_grid.lon.values, iz_grid.lat.values[::-1])
        spline = bisplrep(lons, lats, data)
        gravity_anomaly_0 = bisplev(lonP, latP, spline)

        return gravity_anomaly_0

    def innermost_zone(self, lat, lon):
        """Return Stokes integral value in the innermost zone.

        """

        if interp_method in ('nearest', 'linear'):
            pass
        elif interp_method in ('spline'):
            pass
        else:
            raise ValueError('Not valid interpolation method.')

        dlatr, dlonr = np.radians(self.data.res)
        clat = np.cos(np.radians(lat)))

        s0 = self.ref_radius * np.sqrt(dlonr * dlatr * clat / np.pi)
        return gravity_anomaly * s0

    def innermost_zone(self, gravity_anomaly, lat, method=None):
        """Return Stokes integral in computational point (innermost zone).

        Parameters
        ----------
        gravity_anomaly : float or array_like of floats
            Gravity anomaly.
        lat : float or array_like of floats
            Latitude of the point, in degrees.
        """

        dlatr, dlonr = np.radians(self.data.res)
        clat = np.cos(np.radians(lat)))

        s0 = self.ref_radius * np.sqrt(dlonr * dlatr * clat / np.pi)
        return gravity_anomaly * s0

    def evaluate_point(self, lat, lon, degrees=True):

        lat_bnds = [lat + self.psi2, lat - self.psi2]
        lon_bnds = [lon - self.psi2, lon + self.psi2]
        data_grid = data.sel(lat=slice(*lat_bnds), lon=slice(*lon_bnds))

        dlon, dlat = self.data.res
        dlonr, dlatr = np.radians([dlon, dlat])

        nzn = 4
        gravity_anomaly_0 = self.innermost_interpolation(lat,
                lon, rgrid, nzn, dlat, dlon)

        innermost_zone = self.innermost_zone(
                gravity_anomaly_0, lat, dlatr, dlonr)

        # inner zone
        lonm, latm = np.meshgrid(data_grid.lon.values,
                data_grid.lat.values)

        st_func = self.mean_kernel(lat, lon, latm, lonm, dlat, dlon)
        da = np.radians(dlon) * np.radians(dlat) * np.cos(np.radians(latm))

        inner_zone = np.sum(data_grid.values * st_func * da)
        apot = inner_zone * self._pre_factor + innermost_zone

        return apot

    def evaluate_grid(self, bounds):

        lat_calc_bnds, lon_calc_bnds = bounds[:2], bounds[2:]

        calc_grid = self.data.sel(
                lat=slice(*lat_calc_bnds),
                lon=slice(*lon_calc_bnds))

        dlon, dlat = calc_grid.res
        dlonr, dlatr = np.radians([dlon, dlat])

        # innermost zone
        apot_0 = self.innermost_zone(calc_grid, calc_grid.lat, dlatr, dlonr)

        # inner zone
        inner_zone = np.zeros_like(calc_grid.data)

        lon_point = calc_grid.lon[0].values

        lon_0_bnds = [lon_point - self.psi2, lon_point + self.psi2]

        for jp, latpj in enumerate(tqdm(calc_grid.lat.values)):
            lat_bnds = [latpj + self.psi2, latpj - self.psi2]

            kernel_grid = self.data.sel(
                    lat=slice(*lat_bnds),
                    lon=slice(*lon_0_bnds))

            lonm, latm = np.meshgrid(kernel_grid.lon.values,
                    kernel_grid.lat.values)

            st_func = self.mean_kernel(
                    latpj, lon_point, latm, lonm,
                    dlat, dlon)

            da = dlonr * dlatr * np.cos(np.radians(latm))

            for ip, lonpi in enumerate(calc_grid.lon.values):
                lon_bnds = [lonpi - self.psi2, lonpi + self.psi2]

                data_grid_2 = self.data.sel(
                        lat=slice(*lat_bnds),
                        lon=slice(*lon_bnds))

                inner_zone[jp, ip] = np.sum(data_grid_2.values * st_func * da)

        apot = apot_0 + self._pre_factor * inner_zone

        return apot


# class StokesExtended(SphericalIntegral):
#
#    _name = 'Extended Stokes'
#
#    def kernel(self, radius, spherical_distance, degrees=True):
#        raise NotImplementedError
#
#    def derivative_radius(self, radius, spherical_distance, degrees=True):
#        raise NotImplementedError
#
#    def derivative_spherical_distance(self, radius, spherical_distance, degrees=True):
#        raise NotImplementedError
