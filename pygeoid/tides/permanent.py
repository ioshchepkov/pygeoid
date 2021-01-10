"""Permanent tide.

"""

import numpy as np
import astropy.units as u

from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.constants import (solar_system_gm, g0)
from pygeoid.constants import iers2010

A = -2.9166 * u.m**2 / u.s**2

DEFAULT_FROM_SYSTEM = 'non-tidal'
DEFAULT_TO_SYSTEM = 'mean-tide'


class PermanentTide:
    def __init__(self, m0s0: u.m**2 / u.s**2 = A, a: u.m = 6378137 * u.m,
                 love: dict = iers2010.DEGREE2_LOVE_NUMBERS,
                 gravimetric_factor: float = 1.1563,
                 diminishing_factor: float = 0.6947):

        self.a = a
        self.coeff = m0s0
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
    def potential(self, lat: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return permanent tidal potential.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential : ~astropy.units.Quantity
            Permanent tidal potential.

        """
        return self.coeff * (r / self.a)**2 * (np.sin(lat)**2 - 1 / 3)

    @u.quantity_input
    def potential_lat_derivative(self, lat: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return permanent tidal potential latitude derivative.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential_dlat : ~astropy.units.Quantity
            Permanent Tidal potential latitude derivative.

        """
        return self.coeff * (r / self.a)**2 * np.sin(2 * lat)

    @u.quantity_input
    def potential_r_derivative(self, lat: u.deg, r: u.m) -> u.m / u.s**2:
        """Return permanent tidal potential radial derivative.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential_dr : ~astropy.units.Quantity
            Permanent tidal potential radial derivative.

        """
        return self.coeff * 2 * r / self.a**2 * (np.sin(lat)**2 - 1 / 3)

    @u.quantity_input
    def gradient(self, lat: u.deg, r: u.m) -> u.m / u.s**2:
        """Return permanent tidal potential gradient (tidal acceleration).

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        grad : ~astropy.units.Quantity
            Permanent tidal potential radial derivative.

        """
        r_part = self.potential_r_derivative(lat=lat, r=r)
        lat_part = 1 / r * self.potential_lat_derivative(
            lat=lat, r=r)
        return np.sqrt(r_part**2 + lat_part**2)

    @u.quantity_input
    def displacement(self, lat: u.deg, r: u.m,
                     gravity: u.m / u.s**2 = g0, elastic: bool = True) -> u.m:
        """Return direct tidal deformation for elastic or equilibrium Earth.

        Direct effect is an equipotential surface deformation
        caused by tidal potential. Equilibrium tidal deformation is caused
        only by theoretical tide, elastic deformation takes into account
        the elastic properties of the Earth.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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
            lat, r) * love['l'] / gravity
        u_r = self.potential(
            lat, r) * love['h'] / gravity

        return u.Quantity([u_lat, u_r])

    @u.quantity_input
    def geodetic_height(self, lat: u.deg, r: u.m,
                        gravity: u.m / u.s**2 = g0, elastic: bool = True) -> u.m:
        """Return tidal change in the geodetic height.

        Approximated as the deformation in radial direction.

        Direct effect is an equipotential surface deformation
        caused by tidal potential. Equilibrium tidal deformation is caused
        only by theoretical tide, elastic deformation takes into account
        the elastic properties of the Earth.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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

        return self.potential(lat, r) * love['h'] / gravity

    @u.quantity_input
    def deformation_potential(self, lat: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return deformation permanent tidal potential.

        An incremental (deforamtion) potential is produced by
        the re-distribution of masses which is factorized by a number, the
        second Love number k, on the earth's surface.

        This is also called the indirect effect.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        delta_potential : ~astropy.units.Quantity
            Deformation potentia.

        """
        return self.love['k'] * self.potential_lat_derivative(lat, r)

    @u.quantity_input
    def gravity(self, lat: u.deg, r: u.m,
                elastic: bool = True) -> u.m / u.s**2:
        """Return permanent tidal gravity variation.

        This is just a negative radial derivative of the permanent tidal
        potential.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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
            lat=lat, r=r)

        if elastic:
            radial_derivative *= self.gravimetric_factor

        return -radial_derivative

    @u.quantity_input
    def gravity_correction(self, lat: u.deg, r: u.m,
                           from_system: str = DEFAULT_FROM_SYSTEM,
                           to_system: str = DEFAULT_TO_SYSTEM) -> u.m**2 / u.s**2:
        """Return correction to convert gravity between tide systems.

        Add this value to the gravity in the tide system specified by `from_system`
        argument to get gravity in the tide system specified by `to_system`
        argument.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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

        pot_dr = self.potential_r_derivative(lat, r)

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
    def tilt(self, lat: u.deg, r: u.m,
             azimuth: u.deg = None, gravity: u.m / u.s**2 = g0,
             elastic: bool = True) -> u.dimensionless_unscaled:
        """Return permanent tidal tilt.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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
        xi = -self.potential_lat_derivative(lat=lat, r=r)

        if azimuth is not None:
            g_hor = np.cos(azimuth) * xi
        else:
            g_hor = xi

        tilt = g_hor / (r * gravity)

        if elastic:
            tilt *= -self.diminishing_factor

        return tilt

    @u.quantity_input
    def geoidal_height(self, lat: u.deg, r: u.m,
                       gravity: u.m / u.s**2 = g0) -> u.m:
        """Return permanent tidal variation of the geoidal height.

        Geoidal heights are affected by direct and indirect effects.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_geoid : ~astropy.units.Quantity
            Permanent geoidal height tidal variation.

        """
        potential = self.potential(lat, r)

        return (1 + self.love['k']) / gravity * potential

    @u.quantity_input
    def geoidal_height_correction(self, lat: u.deg, r: u.m,
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
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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

        pot = self.potential(lat, r) / gravity

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
    def physical_height(self, lat: u.deg, r: u.m,
                        gravity: u.m / u.s**2 = g0) -> u.m:
        """Return permanent tidal variation of the (absolute) physical heights.

        Tidal variations of physically defined heights like
        orthometric, normal, or normal-orthometric heights can be
        derived as the difference between ellipsoidal and geoidal heights.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_h : ~astropy.units.Quantity
            Physical heights permanent tidal variation.

        """
        potential = self.potential(lat, r)

        return -self.diminishing_factor / gravity * potential

    @u.quantity_input
    def physical_height_correction(self, lat: u.deg, r: u.m,
                                   from_system: str = DEFAULT_FROM_SYSTEM,
                                   to_system: str = DEFAULT_TO_SYSTEM,
                                   gravity: u.m / u.s**2 = g0) -> u.m:
        """Return correction to convert physical heights between tide systems.

        Add this value to the physical height in the tide system specified by
        `from_system` argument to get physical height in the tide system
        specified by `to_system` argument.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
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

        pot = self.potential(lat, r) / gravity

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

    @u.quantity_input
    def physical_height_difference(self, lat: u.deg, r: u.m,
                                   azimuth: u.deg, length: u.m,
                                   gravity: u.m / u.s**2 = g0) -> u.m:
        """Return permanent tidal variation of the physical heights difference.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
        azimuth : ~astropy.units.Quantity
            Azimuth of the levelling line.
        length : ~astropy.units.Quantity
            Length of the levelling line.
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_h : ~astropy.units.Quantity
            Physical heights difference permanent tidal variation.

        """
        tilt = self.tilt(lat=lat, r=r,
                         azimuth=azimuth, gravity=gravity, elastic=True)

        return length * tilt
