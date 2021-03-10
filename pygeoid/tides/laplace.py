"""Laplace's tidal representation.

"""

import numpy as np
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import (get_body, solar_system_ephemeris, ITRS)

from pygeoid.constants import (solar_system_gm, g0)
from pygeoid.constants import iers2010


def _zonal(lat, declination):
    return 3 * (1 / 3 - np.sin(2 * declination)**2) * (1 / 3 - np.sin(lat)**2)


def _zonal_dlat(lat, declination):
    return 3 * (1 / 3 - np.sin(2 * declination)**2) * (-np.sin(2 * lat))


def _tesseral(lat, declination, hour_angle):
    return np.sin(2 * lat) * np.sin(2 * declination) * np.cos(hour_angle)


def _tesseral_dlat(lat, declination, hour_angle):
    return 2 * np.cos(2 * lat) * np.sin(2 * declination) * np.cos(hour_angle)


def _tesseral_dlon(lat, declination, hour_angle):
    return np.sin(2 * lat) * np.sin(2 * declination) * -np.sin(hour_angle)


def _sectorial(lat, declination, hour_angle):
    return np.cos(lat)**2 * np.cos(declination)**2 * np.cos(2 * hour_angle)


def _sectorial_dlat(lat, declination, hour_angle):
    return -np.sin(2 * lat) * np.cos(declination)**2 * np.cos(2 * hour_angle)


def _sectorial_dlon(lat, declination, hour_angle):
    return np.cos(lat)**2 * np.cos(declination)**2 * -2 * np.sin(2 * hour_angle)


def _doodson(point_distance, body_distance, body_gm):
    return 3 / 4 * body_gm * point_distance**2 / body_distance**3


def _doodson_dr(point_distance, body_distance, body_gm):
    return 3 / 2 * body_gm * point_distance / body_distance**3


_potential_parts = {
    'tesseral': _tesseral,
    'sectorial': _sectorial,
    'zonal': _zonal,
    'doodson' : _doodson
}

_potential_dlat_parts = {
    'tesseral': _tesseral_dlat,
    'sectorial': _sectorial_dlat,
    'zonal': _zonal_dlat,
    'doodson' : _doodson
}

_potential_dlon_parts = {
    'tesseral': _tesseral_dlon,
    'sectorial': _sectorial_dlon,
    'zonal': None,
    'doodson' : _doodson
}

_potential_dr_parts = {
    'tesseral': _tesseral,
    'sectorial': _sectorial,
    'zonal': _zonal,
    'doodson' : _doodson_dr
}

_func_parts = {
    'potential': _potential_parts,
    'potential_dlat': _potential_dlat_parts,
    'potential_dlon': _potential_dlon_parts,
    'potential_dr': _potential_dr_parts
}


class LaplaceTidalEquation:
    """Class for Laplace's tidal representation.

    This class is a realision of the Laplace's tidal representaion for the
    second-degree tidal potential:

        W_2 = D (Z + T + S),

    where D is called Doodson's constant and Z, T and S are zonal, tesseral and
    sectorial  parts of the tidal potential respectively.
    This representation is often applied
    for practical calculations of the total tidal potential and its derivatives,
    instead of more advanced and complicated harmonic decomposition.

    If some tidal effect calculated from this class with all parts included
    (by default) and fot the elastic Earth is applied as a correction
    (effect taken with the negative sign), then the corrected value will be in the conventional
    tide free system, because a permanent part of the tide will also be removed.
    If mean or zero tide systems are desired, then permanent part
    of the tide should be restored in some way.

    Attributes
    ----------
    bodies : list, optional
        List of solar system bodies, the total tide from which will be taken
        into account. By default, only the tides from the Moon and the Sun
        is accounted since they are the largest.
    parts : list, optional
        Which parts of Laplace's tidal eqution are involved. There are 'zonal',
        'tesseral' and 'sectorial' parts. By default, all parts are included.
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
    ephemeris : str, optional
        Ephemeris to use. If not given, use the one set with
        ``astropy.coordinates.solar_system_ephemeris.set`` (which is
        set to 'builtin' by default).

    Notes
    -----
    One can check which bodies are covered by a given ephemeris using:

    >>> from astropy.coordinates import solar_system_ephemeris
    >>> solar_system_ephemeris.bodies

    References
    ----------
    .. [1] Melchior, P. (1983), The Tides of the Planet Earth.
        2nd edition, Pergamon Press.

    """

    def __init__(self, bodies: list = ['moon', 'sun'],
                 parts: list = ['zonal', 'tesseral', 'sectorial'],
                 love: dict = iers2010.DEGREE2_LOVE_NUMBERS,
                 gravimetric_factor: float = 1.1563,
                 diminishing_factor: float = 0.6947,
                 ephemeris: str = None):

        self.bodies = [body.lower() for body in bodies]

        solar_system_ephemeris.set(ephemeris)
        self.ephemeris = ephemeris

        for body in self.bodies:
            if body.lower() == 'pluto':
                raise ValueError('Pluto? Seriosly? No way. '
                                 'It is not even a planet!')
            if body.lower() in ('earth', 'earth-moon-barycenter'):
                raise ValueError(
                    'Do not include \'{0}\' for calculation of the Earth '
                    'tides!'.format(body))
            if body.lower() not in solar_system_ephemeris.bodies:
                raise ValueError('There is no ephemeris for {0}!'.format(body))
            if body.lower() not in solar_system_gm.bodies:
                raise ValueError('There is no GM for {0}!'.format(body))

        for part in parts:
            if part not in ('zonal', 'tesseral', 'sectorial'):
                raise ValueError('There are only zonal, '
                                 'tessseral and sectorial parts!')
        self.parts = parts

        if not all(item in love for item in ('k', 'l', 'h')):
            raise ValueError('All Love ("k", "l", "h") numbers should be '
                             'specified!')

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

        self.bodies_gm = self._get_bodies_gm()

    def _get_bodies(self, time):
        """Get position of the solar system bodies from Astropy.

        """
        return [get_body(body, time, ephemeris=self.ephemeris).
                transform_to(ITRS) for body in self.bodies]

    def _get_bodies_gm(self):
        """Get GM of the solar system bodies.

        """
        return [solar_system_gm.get_body_gm(body) for body in self.bodies]

    def _functional(self, time, lat, lon, r, functional_name):
        """Calculate tidal potenial functionals.

        """
        out = []
        bodies = self._get_bodies(time)
        for idx, body in enumerate(bodies):
            body_gm = self.bodies_gm[idx]
            body_lat = body.spherical.lat
            body_lon = body.spherical.lon
            body_r = body.spherical.distance

            body_out = 0
            if 'tesseral' in self.parts or 'sectorial' in self.parts:
                hour_angle = lon - body_lon
                if 'tesseral' in self.parts:
                    body_out += _func_parts[functional_name]['tesseral'](
                        lat, body_lat, hour_angle)
                if 'sectorial' in self.parts:
                    body_out += _func_parts[functional_name]['sectorial'](
                        lat, body_lat, hour_angle)

            if ('zonal' in self.parts) and (
                    _func_parts[functional_name]['zonal'] is not None):
                body_out += _func_parts[functional_name]['zonal'](
                    lat, body_lat)

            body_out *= _func_parts[
                functional_name]['doodson'](r, body_r, body_gm)
            out.append(body_out)

        return sum(out)

    @u.quantity_input
    def potential(self, time: Time,
                  lat: u.deg, lon: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return tidal potential.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential : ~astropy.units.Quantity
            Tidal potential.

        """
        return self._functional(time=time, lat=lat, lon=lon, r=r,
                                functional_name='potential')

    @u.quantity_input
    def potential_lat_derivative(self, time: Time,
                                 lat: u.deg, lon: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return tidal potential latitude derivative.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential_dlat : ~astropy.units.Quantity
            Tidal potential latitude derivative.

        """
        return self._functional(time=time, lat=lat, lon=lon, r=r,
                                functional_name='potential_dlat')

    @u.quantity_input
    def potential_lon_derivative(self, time: Time,
                                 lat: u.deg, lon: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return tidal potential longitude derivative.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential_dlon : ~astropy.units.Quantity
            Tidal potential longitude derivative.

        """
        return self._functional(time=time, lat=lat, lon=lon, r=r,
                                functional_name='potential_dlon')

    @u.quantity_input
    def potential_r_derivative(self, time: Time,
                               lat: u.deg, lon: u.deg, r: u.m) -> u.m / u.s**2:
        """Return tidal potential radial derivative.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        potential_dr : ~astropy.units.Quantity
            Tidal potential radial derivative.

        """
        return self._functional(time=time, lat=lat, lon=lon, r=r,
                                functional_name='potential_dr')

    @u.quantity_input
    def gradient(self, time: Time,
                 lat: u.deg, lon: u.deg, r: u.m) -> u.m / u.s**2:
        """Return tidal potential gradient (tidal acceleration).

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        grad : ~astropy.units.Quantity
            Tidal potential radial derivative.

        """
        r_part = self.potential_r_derivative(time=time, lat=lat, lon=lon, r=r)
        lat_part = 1 / r * self.potential_lat_derivative(
            time=time, lat=lat, lon=lon, r=r)
        lon_part = 1 / (r * np.cos(lat)) * self.potential_lat_derivative(
            time=time, lat=lat, lon=lon, r=r)

        return np.sqrt(r_part**2 + lat_part**2 + lon_part**2)

    @u.quantity_input
    def displacement(self, time: Time, lat: u.deg, lon: u.deg, r: u.m,
                     gravity: u.m / u.s**2 = g0, elastic: bool = True) -> u.m:
        """Return direct tidal deformation for elastic or equilibrium Earth.

        Direct effect is an equipotential surface deformation
        caused by tidal potential. Equilibrium tidal deformation is caused
        only by theoretical tide, elastic deformation takes into account
        the elastic properties of the Earth.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
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
        u_lon : ~astropy.units.Quantity
            Deformation in the east-west direction.
        u_r : ~astropy.units.Quantity
            Deformation in the radial direction.

        """
        if elastic:
            love = self.love
        else:
            love = {'l' : 1.0, 'h' : 1.0}

        u_lat = self.potential_lat_derivative(
            time, lat, lon, r) * love['l'] / gravity
        u_lon = self.potential_lon_derivative(
            time, lat, lon, r) * love['l'] / (gravity * np.cos(lat))
        u_r = self.potential(time, lat, lon, r) * love['h'] / gravity

        return u.Quantity([u_lat, u_lon, u_r])

    @u.quantity_input
    def deformation_potential(self, time: Time,
                              lat: u.deg, lon: u.deg, r: u.m) -> u.m**2 / u.s**2:
        """Return deformation tidal potential.

        An incremental (deforamtion) potential is produced by
        the re-distribution of masses which is factorized by a number, the
        second Love number k, on the earth's surface.

        This is also called the indirect effect.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).

        Returns
        -------
        delta_potential : ~astropy.units.Quantity
            Deformation potential.

        """
        potential = self.potential_lat_derivative(
            time, lat, lon, r)

        return potential * self.love['k']

    @u.quantity_input
    def gravity(self, time: Time,
                lat: u.deg, lon: u.deg, r: u.m,
                elastic: bool = True) -> u.m / u.s**2:
        """Return tidal gravity variation.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
        elastic : bool, optional
            If True then the Earth is elastic (deformable)
            and gravity change is multiplied by the gravimetric factor
            specified in the class instance.

        Returns
        -------
        delta_g : ~astropy.units.Quantity
            Tidal gravity variation.

        """
        radial_derivative = self.potential_r_derivative(
            time=time, lat=lat, lon=lon, r=r)

        if elastic:
            radial_derivative *= self.gravimetric_factor

        return -radial_derivative

    @u.quantity_input
    def tilt(self, time: Time, lat: u.deg, lon: u.deg, r: u.m,
             azimuth: u.deg = None, gravity: u.m / u.s**2 = g0,
             elastic: bool = True) -> u.dimensionless_unscaled:
        """Return tidal tilt.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
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
            Tidal tilt.

        """
        xi = -self.potential_lat_derivative(
            time=time, lat=lat, lon=lon, r=r)
        eta = -1 / np.cos(lat) * self.potential_lon_derivative(
            time=time, lat=lat, lon=lon, r=r)

        if azimuth is not None:
            g_hor = (np.cos(azimuth) * xi + np.sin(azimuth) * eta)
        else:
            g_hor = np.sqrt(xi**2 + eta**2)

        tilt = g_hor / (r * gravity)

        if elastic:
            tilt *= -self.diminishing_factor

        return tilt

    @u.quantity_input
    def geoidal_height(self, time: Time,
                       lat: u.deg, lon: u.deg, r: u.m,
                       gravity: u.m / u.s**2 = g0) -> u.m:
        """Return tidal variation of the geoidal height.

        Geoidal heights are affected by direct and indirect effects.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_geoid : ~astropy.units.Quantity
            Geoidal height tidal variation.

        """
        potential = self.potential(
            time, lat, lon, r)

        return (1 + self.love['k']) / gravity * potential

    @u.quantity_input
    def physical_height(self, time: Time,
                        lat: u.deg, lon: u.deg, r: u.m,
                        gravity: u.m / u.s**2 = g0) -> u.m:
        """Return tidal variation of the (absolute) physical heights.

        Tidal variations of physically defined heights like
        orthometric, normal, or normal-orthometric heights can be
        derived as the difference between ellipsoidal and geoidal heights.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
        r   : ~astropy.units.Quantity
            Geocentric radius (radial distance).
        gravity : ~astropy.units.Quantity, optional
            Mean global gravity of the Earth.

        Returns
        -------
        delta_h : ~astropy.units.Quantity
            Physical heights tidal variation.

        """
        potential = self.potential(
            time, lat, lon, r)

        return -self.diminishing_factor / gravity * potential

    @u.quantity_input
    def physical_height_difference(self, time: Time,
                                   lat: u.deg, lon: u.deg, r: u.m,
                                   azimuth: u.deg, length: u.m,
                                   gravity: u.m / u.s**2 = g0) -> u.m:
        """Return tidal variation of the physical heights difference.

        Parameters
        ----------
        time : ~astropy.time.Time
            Time of observation.
        lat : ~astropy.units.Quantity
            Geocentric (spherical) latitude.
        lon : ~astropy.units.Quantity
            Geocentric (spherical) longitude.
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
            Physical heights difference tidal variation.

        """
        tilt = self.tilt(time=time, lat=lat, lon=lon, r=r,
                         azimuth=azimuth, gravity=gravity, elastic=True)

        return -length * tilt
