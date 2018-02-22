"""
Gravity field and geometry of the level ellipsoid.
"""

import numpy as np
from scipy import (optimize, special)
from pygeoid.coordinates.ellipsoid import Ellipsoid


_pj_ellps = {
    'GRS80': {'description': 'GRS 80', 'a': 6378137.0, 'j2': 108263e-8,
              'gm': 3986005e8, 'omega': 7292115e-11},
    'WGS84': {'description': 'WGS 84', 'a': 6378137.0, 'rf': 298.2572235630,
              'gm': 3986004.418e8, 'omega': 7292115e-11},
    'PZ90': {'description': 'PZ 90.11', 'a': 6378136.0, 'rf': 298.25784,
             'gm': 3986004.418e8, 'omega': 7292115e-11},
    'GSK2011': {'description': 'GSK-2011', 'a': 6378136.5, 'rf': 298.2564151,
                'gm': 3986004.415e8, 'omega': 7292115e-11},
}

# default level ellipsoid for normal gravity field
DEFAULT_LEVEL_ELLIPSOID = 'GRS80'


def _j2_to_flattening(j2, a, gm, omega):
    """Calculate flattening from J2, a, GM and omega.

    """
    m1 = omega**2 * a**3 / gm

    def e2(e2, j2, m1):
        e1 = np.sqrt(e2 / (1 - e2))
        q0 = 0.5*((1 + 3 / e1**2) * np.arctan(e1) - 3 / e1)
        return 3*j2 + 2/15 * m1 * np.sqrt(e2)**3 / q0 - e2

    _e2_0 = 3 * j2 + 2/15 * m1
    _e2 = optimize.fsolve(e2, _e2_0, args=(j2, m1), xtol=1e-10)[0]

    return 1 - np.sqrt(1 - _e2)


class LevelEllipsoid(Ellipsoid):
    """Class represents the gravity field of the level ellipsoid.

    This class intialize `Ellipsoid` class from `pygeoid.coordinates.ellipsoid`, so
    all geometrical methods and parameters are available too.

    Parameters
    ----------
    ellps : {'GRS80', 'WGS84', 'PZ90', 'GSK2011'}, optional
        Ellipsoid name. Default is 'GRS80'.
    """
    # pylint: disable=R0904

    def __init__(self, ellps=None, **kwargs):
        if not kwargs:
            if ellps in _pj_ellps:
                kwargs = _pj_ellps[ellps]
            elif ellps is None or ellps.lower() == 'default':
                kwargs = _pj_ellps[DEFAULT_LEVEL_ELLIPSOID]
            else:
                raise ValueError(
                    'No ellipsoid with name {:%s}, possible values \
                        are:\n{:%s}'.format(ellps,
                                            _pj_ellps.keys()))

        if 'j2' in kwargs:
            kwargs['f'] = _j2_to_flattening(kwargs['j2'], kwargs['a'],
                                            kwargs['gm'], kwargs['omega'])
            kwargs['_j2'] = kwargs.pop('j2')

        super().__init__(self, **kwargs)

        self.gm = kwargs.pop('gm')
        self.omega = kwargs.pop('omega')

        # define useful short-named attributes
        # pylint: disable=C0103
        self.m = self.omega**2 * self.a**2 * self.b / self.gm
        self._q0 = 0.5*((1 + 3 / self.e1**2) *
                        np.arctan(self.e1) - 3 / self.e1)

        if not hasattr(self, '_j2'):
            self._j2 = self.e2 / 3 * (1 - 2/15*self.m*self.e1/self._q0)

        self._q01 = 3 * (1 + 1 / self.e12) * (1 -
                                              np.arctan(self.e1) / self.e1) - 1

        self._surface_potential = self.gm / self.linear_eccentricity *\
            np.arctan(self.second_eccentricity) +\
            1 / 3 * self.omega ** 2 * self.a ** 2

        self._gamma_e = self.gm / (self.a * self.b) * (1 -
                                                       self.m - self.m / 6 *
                                                       self.e1 * self._q01 / self._q0)

        self._gamma_p = self.gm / self.a**2 * (1 +
                                               self.m / 3 * self.e1 * self._q01 /
                                               self._q0)
        self._gravity_flattening = (self._gamma_p - self._gamma_e) /\
            self._gamma_e

        self._k = (self.b * self._gamma_p - self.a * self._gamma_e) / (self.a *
                                                                       self._gamma_e)

    @property
    def j2(self):
        """Return dynamic form factor J2.

        """
        return self._j2

    @property
    def surface_potential(self):
        """Return normal gravity potential on the ellipsoid, in m**2/s**2.

        Value of the normal gravity potential on the ellipsoid, or on the
        equipotential surface U(x, y, z) = U_0
        """
        return self._surface_potential

    def _q(self, u):
        """Return auxiliary function q(u).

        """
        E = self.linear_eccentricity
        return 0.5 * ((1 + 3 * u**2 / E**2) * np.arctan2(E, u) - 3 * u / E)

    def gravitational_potential(self, rlat, u, degrees=True):
        """Return normal gravitational potential V, in m**2/s**2.

        Calculate normal gravitational potential from the rigorous formula.

        Parameters
        ----------
        rlat : float or array_like of floats
            Reduced latitude.
        u : float or array_like of floats
            Polar axis of the ellipsoid passing through the point.
        degrees : bool, optional
            If True, the input `rlat` is given in degrees,
            otherwise radians.

        Returns
        -------
        float or array_like of floats
            Normal gravitational potential, in m/s**2.
        """

        if degrees:
            rlat = np.radians(rlat)

        E = self.linear_eccentricity
        arctanEu = np.arctan2(E, u)
        _qr = self._q(u) / self._q0

        return (self.gm / E) * arctanEu + \
            0.5 * self.omega**2 * self.a**2 * \
            _qr * (np.sin(rlat)**2 - 1 / 3)

    def gravity_potential(self, rlat, u, degrees=True):
        """Return normal gravity potential U, in m**2/s**2.

        Calculate normal gravity potential from the rigorous formula.

        Parameters
        ----------
        rlat : float or array_like of floats
            Reduced latitude.
        u : float or array_like of floats
            Polar axis of the ellipsoid passing through the point.
        degrees : bool, optional
            If True, the input `rlat` is given in degrees,
            otherwise radians.

        Returns
        -------
        float or array_like of floats
            Normal gravity potential, in m**2/s**2.
        """

        if degrees:
            rlat = np.radians(rlat)

        gravitational = self.gravitational_potential(rlat, u, degrees=False)
        centrifugal = 0.5 * self.omega**2 * (u**2 +
                                             self.linear_eccentricity**2) * np.cos(rlat)**2
        return gravitational + centrifugal

    @property
    def equatorial_normal_gravity(self):
        """Return normal gravity at the equator, in m/s**2.

        """
        return self._gamma_e

    @property
    def polar_normal_gravity(self):
        """Return normal gravity at the poles, in m/s**2.

        """
        return self._gamma_p

    @property
    def gravity_flattening(self):
        """Return gravity flattening.

        f* = (gamma_p - gamma_e) / gamma_e
        """
        return self._gravity_flattening

    def conventional_gravity_coeffs(self):
        """Return coefficients for the conventional gravity formula.

        """
        f4 = -0.5 * self.f**2 + 2.5 * self.f * self.m * self.polar_radius / self.a
        return (self.gravity_flattening, 0.25 * f4)

    def surface_normal_gravity(self, lat, degrees=True):
        """Return normal gravity on the ellipsoid, in m/s**2.

        Calculate normal gravity value on the level ellipsoid by the rigorous
        formula of Somigliana.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.
        degrees : bool, optional
            If True, the input `lat` is given in degrees, otherwise radians.
            Default is True.

        Returns
        -------
        float or array_like of floats
            Normal gravity on the ellipsoid, in m/s**2.
        """
        if degrees:
            lat = np.radians(lat)

        return self._gamma_e * (1 + self._k * np.sin(lat) ** 2) / self._w(lat)

    def surface_vertical_normal_gravity_gradient(self, lat, degrees=True):
        """Return the vertical gravity gradient on the ellipsoid, in 1/s**2.

        Vertical gradient of the normal gravity at the reference ellipsoid.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.
        degrees : bool, optional
            If True, the input `lat` is given in degrees, otherwise radians.
            Default is True.
        """
        if degrees:
            lat = np.radians(lat)

        gamma = self.surface_normal_gravity(lat, degrees=False)
        return -2 * gamma * self.average_curvature(lat, degrees=False) -\
            2 * self.omega ** 2

    def normal_gravity(self, rlat, u, degrees=True):
        """Return normal gravity, in m/s**2.

        Calculate normal gravity value at any arbitrary point by the rigorous
        closed formula.

        Parameters
        ----------
        rlat : float or array_like of floats
            Reduced latitude.
        u : float or array_like of floats
            Polar axis of the ellipsoid passing through the point.
        degrees : bool, optional
            If True, the input `rlat` is given in degrees,
            otherwise radians.

        Returns
        -------
        float or array_like of floats
            Normal gravity, in m/s**2.
        """

        if degrees:
            rlat = np.radians(rlat)

        E = self.linear_eccentricity
        _qr = self._q(u) / self._q0

        uE = u**2 + E**2
        w = np.sqrt((u**2 + E**2 * np.sin(rlat)**2) / uE)
        q1 = 3 * (1 + u**2 / E**2) * (1 - u / E * np.arctan2(E, u)) - 1

        yu = self.gm / uE
        yu += self.omega**2 * self.a ** 2 * E / uE * q1 / self._q0 * (
            0.5 * np.sin(rlat)**2 - 1 / 6)
        yu -= self.omega**2 * u * np.cos(rlat)**2
        yu *= -1 / w

        yrlat = -self.omega**2 * self.a**2 / np.sqrt(uE) * _qr
        yrlat += self.omega**2 * np.sqrt(uE)
        yrlat *= -1 / w * np.sin(rlat) * np.cos(rlat)

        return np.sqrt(yu**2 + yrlat**2)

    def j2n(self, n):
        """Return even zonal coefficients J with a degree of 2*n.

        If n = 0, the function will return -1.
        If n = 1, the function will return J2 (dynamic form factor).
        If n = 2, the function will return J4.
        If n = 3, the function will return J6.
        If n = 4, the function will return J8.

        Parameters
        ----------
        n : int
            Degree of the J coefficient.
        """
        j2n = (-1)**(n + 1) * (3 * self.e2**(n - 1)) / ((2 * n + 1) * (2 * n + 3)) * \
            ((1 - n) * self.e2 + 5 * n * self.j2)
        return j2n

    def gravitational_potential_sph(self, lat, radius, n_max=4, degrees=True):
        """Return normal gravitational potential V, in m**2/s**2.

        Calculate normal gravitational potential from spherical approximation.

        Parameters
        ----------
        lat : float or array_like of floats
            Spherical (geocentric) latitude.
        radius : float or array_like of floats
            Radius, in metres.
        n_max : int
            Maximum degree.
        degrees : bool, optional
            If True, the input `lat` is given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = np.radians(lat)

        Vns = 0
        for l in range(1, n_max + 1):
            leg = special.eval_legendre(2 * l, np.sin(lat))
            Vns += self.j2n(l) * (self.a / radius) ** (2 * l) * leg
        return self.gm / radius * (1 - Vns)

    def gravity_potential_sph(self, lat, radius, degrees=True, **kwargs):
        """Return normal gravitational potential V, in m**2/s**2.

        Calculate normal gravitational potential from spherical approximation.

        Parameters
        ----------
        lat : float or array_like of floats
            Spherical (geocentric) latitude.
        radius : float or array_like of floats
            Radius, in metres.
        n_max : int
            Maximum degree.
        degrees : bool, optional
            If True, the input `lat` is given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = np.radians(lat)

        gravitational_sph = self.gravitational_potential_sph(lat=lat,
                                                             radius=radius, degrees=False, **kwargs)
        centrifugal = 0.5 * self.omega**2 * radius**2 * np.cos(lat)**2
        return gravitational_sph + centrifugal


normal_gravity_coeffs = {
    'helmert': (978030, 0.005302, 0.000007),
    'helmert_14mGal': (978030 - 14, 0.005302, 0.000007),
    '1930': (978049, 0.0052884, 0.0000059),
    '1930_14mGal': (978049 - 14, 0.0052884, 0.0000059),
    '1967': (978031.8, 0.0053024, 0.0000059),
    '1980': (978032.7, 0.0053024, 0.0000058)}


def surface_normal_gravity_clairaut(lat, model=None, degrees=True):
    """Return normal gravity from the first Clairaut formula, in mGal.

    Parameters
    ----------
    lat : float or array_like of floats
        Geodetic latitude.
    model : {'helmert', 'helmert_14mGal', '1930',\
            '1930_14mGal', '1967', '1980'}
        Which gravity formula will be used.
    degrees : bool, optional
        If True, the input `lat` is given in degrees, otherwise radians.
        Default is True.
    """

    if model is not None and model in normal_gravity_coeffs:
        gamma_e, beta, beta1 = normal_gravity_coeffs[model]
    else:
        msg = 'No formula with name {:%s}, possible values are:\n{:%s}'
        raise ValueError(msg.format(model, model.keys()))

    if degrees:
        lat = np.radians(lat)

    return gamma_e * (1 + beta * np.sin(lat)**2 - beta1 * np.sin(2*lat)**2)
