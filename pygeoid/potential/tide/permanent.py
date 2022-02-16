"""Permanent tide.

"""

import numpy as np
import astropy.units as u

from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.transform import geodetic_to_cartesian
from pygeoid.potential.core import PotentialBase as _PotentialBase

from pygeoid.constants import (solar_system_gm, g0)
from pygeoid.constants import iers2010

__all__ = ['PermanentTide']

DEFAULT_COEFF_A = -2.9166 * u.m**2 / u.s**2

DEFAULT_FROM_SYSTEM = 'non-tidal'
DEFAULT_TO_SYSTEM = 'mean-tide'


class PermanentTide(_PotentialBase):
    """Class for permanent part of the tide-generating potential M0S0.

    The permanent part of the tide-generating potential can be written as

    M0S0 = A * (r / r0)**2 * (sin(lat)**2 - 1/3)

    where lat is the geocentric latitude, r is the rasial distance of the
    observation point and r0 a scaling factor for distances.

    With the normalization r0 = 6378136.3 m used in the tidal
    potential coefficients in Hartmann and Wenzel (1995) which
    is commonly applied in software for tidal analysis and prediction,
    the value A = -2.9166 m**2 / s**2 in the epoch 2000.0
    can be deduced from the IERS Conventions (2010).

    Attributes
    ----------
    coeff   : ~astropy.units.Quantity, optional
        Coefficient A for M0S0 in given representaion.
    r0   : ~astropy.units.Quantity, optional
        Scaling factor for distance.
    love : dict, optional
        Approximate Love numbers for elastic deformation calculations. This
        dictionary should contain 'k', 'l' and 'h' numbers of the second degree.
        Default values are taken from IERS Conventions 2010.
    gravimetric_factor : float, optional
        Gravimetric factor (delta). If None it will be calculated from Love
        numbers as delta = 1 + h - 3/2*k. Default value is approximate gravimetric
        factor from PREM model.
    diminishing_factor : float, optional
        Diminishing factor (gamma). If None then it will be calculated from Love
        numbers as gamma = 1 + k - h. Default value is approximate diminishing
        factor from PREM model.

    References
    ----------
    .. [1] Mäkinen J (2020, submitted) The permanent tide and the
        International Height Reference Frame IHRF. Journal of Geodesy
        arXiv:2006.08440

    """

    def __init__(self, coeff: u.m**2 / u.s**2 = DEFAULT_COEFF_A,
                 r0: u.m = 6378137 * u.m,
                 love: dict = iers2010.DEGREE2_LOVE_NUMBERS,
                 gravimetric_factor: float = 1.1563,
                 diminishing_factor: float = 0.6947):

        self.coeff = coeff
        self.r0 = r0
        self.love = love

        if gravimetric_factor is None:
            self.gravimetric_factor = 1 + self.love['h'] - \
                3 / 2 * self.love['k']
        else:
            self.gravimetric_factor = gravimetric_factor

        if diminishing_factor is None:
            self.diminishing_factor = 1 + self.love['k'] - self.love['h']
        else:
            self.diminishing_factor = diminishing_factor

    @u.quantity_input
    def _potential(self, position) -> u.m**2 / u.s**2:
        """Return permanent tidal potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        potential : ~astropy.units.Quantity
            Permanent tidal potential.

        """
        sph = position.represent_as('spherical')
        return self.coeff * (sph.distance / self.r0)**2 * (
            np.sin(sph.lat)**2 - 1 / 3)

    def _derivative_spherical(self, position, variable):
        if variable in ('lat', 'latitude'):
            return self.potential_lat_derivative(position)
        elif variable in ('lon', 'longitude', 'long'):
            return 0.0 * u.m**2 / u.s**2 / u.rad
        elif variable in ('distance', 'radius', 'r', 'radial'):
            return self.potential_r_derivative(position)

    @u.quantity_input
    def potential_lat_derivative(self, position) -> u.m**2 / u.s**2 / u.rad:
        """Return permanent tidal potential latitude derivative.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        potential_dlat : ~astropy.units.Quantity
            Permanent Tidal potential latitude derivative.

        """
        sph = position.represent_as('spherical')
        return self.coeff * (
            sph.distance / self.r0)**2 * np.sin(2 * sph.lat) / u.rad

    @u.quantity_input
    def potential_r_derivative(self, position) -> u.m / u.s**2:
        """Return permanent tidal potential radial derivative.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        potential_dr : ~astropy.units.Quantity
            Permanent tidal potential radial derivative.

        """
        sph = position.represent_as('spherical')
        return (self.coeff * 2 * sph.distance / self.r0**2 * (
            np.sin(sph.lat)**2 - 1 / 3))

    @u.quantity_input
    def displacement(self, position,
                     gravity: u.m / u.s**2 = g0, elastic: bool = True) -> u.m:
        """Return direct tidal deformation for elastic or equilibrium Earth.

        Direct effect is an equipotential surface deformation
        caused by tidal potential. Equilibrium tidal deformation is caused
        only by theoretical tide, elastic deformation takes into account
        the elastic properties of the Earth.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        gravity : ~astropy.units.Quantity
            Mean global gravity of the Earth.
        elastic : bool, optional
            If True, elastic deformation (with approximate Love numbers)
            will be returned, otherwise equilibrium deformation.
            Default is True.

        Returns
        -------
        u_lat : ~astropy.units.Quantity
            Deformation in the north-south direction.
        u_r : ~astropy.units.Quantity
            Deformation in the radial direction.

        """
        if elastic:
            love = self.love
        else:
            love = {'l' : 1.0, 'h' : 1.0}

        u_lat = self.potential_lat_derivative(
            position) * love['l'] / gravity * u.rad
        u_r = self.potential(
            position) * love['h'] / gravity

        return u.Quantity([u_lat, u_r])

    @u.quantity_input
    def geodetic_height(self, position,
                        gravity: u.m / u.s**2 = g0, elastic: bool = True) -> u.m:
        """Return tidal change in the geodetic height.

        Approximated as the deformation in radial direction.

        Direct effect is an equipotential surface deformation
        caused by tidal potential. Equilibrium tidal deformation is caused
        only by theoretical tide, elastic deformation takes into account
        the elastic properties of the Earth.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        gravity : ~astropy.units.Quantity
            Mean global gravity of the Earth.
        elastic : bool, optional
            If True, elastic deformation (with approximate Love numbers)
            will be returned, otherwise equilibrium deformation.
            Default is True.

        Returns
        -------
        delta_h : ~astropy.units.Quantity
            Tidal variation of the earth's surface geodetic height.

        """
        if elastic:
            love = self.love
        else:
            love = {'l' : 1.0, 'h' : 1.0}

        return self.potential(position) * love['h'] / gravity

    @u.quantity_input
    def deformation_potential(self, position) -> u.m**2 / u.s**2:
        """Return deformation permanent tidal potential.

        An incremental (deforamtion) potential is produced by
        the re-distribution of masses which is factorized by a number, the
        second Love number k, on the earth's surface.

        This is also called the indirect effect.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        delta_potential : ~astropy.units.Quantity
            Deformation potentia.

        """
        return self.love['k'] * self.potential_lat_derivative(
            position) * u.rad

    @u.quantity_input
    def gravity(self, position,
                elastic: bool = True) -> u.m / u.s**2:
        """Return permanent tidal gravity variation.

        This is just a negative radial derivative of the permanent tidal
        potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        elastic : bool, optional
            If True then the Earth is elastic (deformable)
            and gravity change is multiplied by the gravimetric factor
            specified in the class instance.

        Returns
        -------
        delta_g : ~astropy.units.Quantity
            Permanent Tidal gravity variation.

        """
        radial_derivative = self.potential_r_derivative(
            position)

        if elastic:
            radial_derivative *= self.gravimetric_factor

        return -radial_derivative

    @u.quantity_input
    def gravity_ell(self, position,
                    elastic: bool = True) -> u.m / u.s**2:
        """Return permanent tidal gravity variation along the ellipsoidal normal.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        elastic : bool, optional
            If True then the Earth is elastic (deformable)
            and gravity change is multiplied by the gravimetric factor
            specified in the class instance.

        Returns
        -------
        delta_g : ~astropy.units.Quantity
            Permanent tidal potential gravity variation.

        """
        ell = position._ellipsoid
        geod = position.geodetic
        pvcr = ell.prime_vertical_curvature_radius(geod.lat)

        delta_g = 2 / 3 * self.coeff / self.r0**2 * (
            (pvcr * (3 - 2 * ell.e2) + 3 * geod.height) * np.sin(geod.lat)**2 -
            (pvcr + geod.height))

        if elastic:
            delta_g *= self.gravimetric_factor

        return -delta_g

    @u.quantity_input
    def convert_gravity_correction(self, position,
                                   from_system: str = DEFAULT_FROM_SYSTEM,
                                   to_system: str = DEFAULT_TO_SYSTEM) -> u.m**2 / u.s**2:
        """Return correction to convert gravity between tide systems.

        Add this value to the gravity in the tide system specified by `from_system`
        argument to get gravity in the tide system specified by `to_system`
        argument.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        from_system : {'zero-tide', 'mean-tide', 'non-tidal'}, optional
            In which tide system the gravity is given. Default value is set to
            `pygeoid.tides.permanent.DEFAULT_FROM_SYSTEM`.
        to_system : {'zero-tide', 'mean-tide', 'non-tidal'}, optional
            In which tide system is the gravity converted. Default value is set to
            `pygeoid.tides.permanent.DEFAULT_TO_SYSTEM`.

        Returns
        -------
        delta_g : ~astropy.units.Quantity
            Correction to the gravity value.

        """

        pot_dr = self.potential_r_derivative(position)

        if from_system == 'zero-tide' and to_system == 'mean-tide':
            factor = 1
        elif from_system == 'mean-tide' and to_system == 'zero-tide':
            factor = -1
        elif from_system == 'mean-tide' and to_system == 'non-tidal':
            factor = -self.gravimetric_factor
        elif from_system == 'non-tidal' and to_system == 'mean-tide':
            factor = self.gravimetric_factor
        elif from_system == 'zero-tide' and to_system == 'non-tidal':
            factor = -(self.gravimetric_factor - 1)
        elif from_system == 'non-tidal' and to_system == 'zero-tide':
            factor = (self.gravimetric_factor - 1)

        return factor * pot_dr

    @u.quantity_input
    def tilt(self, position,
             azimuth: u.deg = None, gravity: u.m / u.s**2 = g0,
             elastic: bool = True) -> u.dimensionless_unscaled:
        """Return permanent tidal tilt.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        azimuth : ~astropy.units.Quantity, optional
            If given then tilt will be returned in that direction.
            If None then full tilt angle will be returned.
            Default is None.
        elastic : bool, optional
            If True then the Earth is elastic (deformable)
            and tilt is multiplied by the combination of Love
            numbers (h - (1 + k)).
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        tilt : ~astropy.units.Quantity
            Permanent tidal tilt.

        """
        xi = -self.potential_lat_derivative(position) * u.rad

        if azimuth is not None:
            g_hor = np.cos(azimuth) * xi
        else:
            g_hor = xi

        tilt = g_hor / (position.spherical.distance * gravity)

        if elastic:
            tilt *= -self.diminishing_factor

        return tilt

    @u.quantity_input
    def geoidal_height(self, position,
                       gravity: u.m / u.s**2 = g0) -> u.m:
        """Return permanent tidal variation of the geoidal height.

        Geoidal heights are affected by direct and indirect effects.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_geoid : ~astropy.units.Quantity
            Permanent geoidal height tidal variation.

        """
        potential = self.potential(position)

        return (1 + self.love['k']) / gravity * potential

    @u.quantity_input
    def convert_geoidal_height_correction(self, position,
                                          from_system: str = DEFAULT_FROM_SYSTEM,
                                          to_system: str = DEFAULT_TO_SYSTEM,
                                          gravity: u.m / u.s**2 = g0) -> u.m:
        """Return correction to convert geoidal height between tide systems.

        Add this value to the geoidal height (or height anomaly)
        in the tide system specified by `from_system`
        argument to get geoidal height (or height anomaly)
        in the tide system specified by `to_system`
        argument.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        from_system : {'zero-tide', 'mean-tide', 'non-tidal'}, optional
            In which tide system the geoidal height is given.
            Default value is set to `pygeoid.tides.permanent.DEFAULT_FROM_SYSTEM`.
        to_system : {'zero-tide', 'mean-tide', 'non-tidal'}, optional
            In which tide system is the geoidal height converted.
            Default value is set to `pygeoid.tides.permanent.DEFAULT_TO_SYSTEM`.
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_zeta : ~astropy.units.Quantity
            Correction to the geoidal height.

        """

        pot = self.potential(position) / gravity

        if from_system == 'zero-tide' and to_system == 'mean-tide':
            factor = 1
        elif from_system == 'mean-tide' and to_system == 'zero-tide':
            factor = -1
        elif from_system == 'mean-tide' and to_system == 'non-tidal':
            factor = -(self.love['k'] + 1)
        elif from_system == 'non-tidal' and to_system == 'mean-tide':
            factor = (self.love['k'] + 1)
        elif from_system == 'zero-tide' and to_system == 'non-tidal':
            factor = -self.love['k']
        elif from_system == 'non-tidal' and to_system == 'zero-tide':
            factor = self.love['k']

        return factor * pot

    @u.quantity_input
    def physical_height(self, position,
                        gravity: u.m / u.s**2 = g0) -> u.m:
        """Return permanent tidal variation of the (absolute) physical heights.

        Tidal variations of physically defined heights like
        orthometric, normal, or normal-orthometric heights can be
        derived as the difference between ellipsoidal and geoidal heights.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_h : ~astropy.units.Quantity
            Physical heights permanent tidal variation.

        """
        potential = self.potential(position)

        return -self.diminishing_factor / gravity * potential

    @u.quantity_input
    def convert_physical_height_correction(self, position,
                                           from_system: str = DEFAULT_FROM_SYSTEM,
                                           to_system: str = DEFAULT_TO_SYSTEM,
                                           gravity: u.m / u.s**2 = g0) -> u.m:
        """Return correction to convert physical heights between tide systems.

        Add this value to the physical height in the tide system specified by
        `from_system` argument to get physical height in the tide system
        specified by `to_system` argument.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        from_system : {'zero-tide', 'mean-tide', 'non-tidal'}, optional
            In which tide system the pgysical height is given.
            Default value is set to `pygeoid.tides.permanent.DEFAULT_FROM_SYSTEM`.
        to_system : {'zero-tide', 'mean-tide', 'non-tidal'}, optional
            In which tide system is the physical height converted.
            Default value is set to
            `pygeoid.tides.permanent.DEFAULT_TO_SYSTEM`.
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_h : ~astropy.units.Quantity
            Correction to the physical height value.

        """

        pot = self.potential(position) / gravity

        if from_system == 'zero-tide' and to_system == 'mean-tide':
            factor = 1
        elif from_system == 'mean-tide' and to_system == 'zero-tide':
            factor = -1
        elif from_system == 'mean-tide' and to_system == 'non-tidal':
            factor = -self.diminishing_factor
        elif from_system == 'non-tidal' and to_system == 'mean-tide':
            factor = self.diminishing_factor
        elif from_system == 'zero-tide' and to_system == 'non-tidal':
            factor = -(self.diminishing_factor - 1)
        elif from_system == 'non-tidal' and to_system == 'zero-tide':
            factor = (self.diminishing_factor - 1)

        return factor * pot
