"""
Gravity field and geometry of the level ellipsoid.
"""

import numpy as np
from scipy import (optimize, special)
from pygeoid.coordinates.ellipsoid import Ellipsoid


LEVEL_ELLIPSOIDS = {
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


class Centrifugal:
    """Centrifugal potential and its derivatives.

    Parameters
    ----------
    omega : float
        Angular rotation rate of the body, in rad/s.
        Default value is the angular speed of
        Earth's rotation 7292115e-11 rad/s
    """

    def __init__(self, omega=7292115e-11):
        self.omega = omega

    def potential(self, lat, radius, degrees=True):
        """Return centrifugal potential, in m**2/s**2.

        Parameters
        ----------
        lat : float or array_like of floats
            Spherical (geocentric) latitude.
        radius : float or array_like of floats
            Radius, in metres.
        degrees : bool, optional
            If True, the input `lat` is given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = np.radians(lat)

        return 0.5 * self.omega**2 * radius**2 * np.cos(lat)**2

    def r_derivative(self, lat, radius, degrees=True):
        """Return radial derivative, in 1/s**2.

        Parameters
        ----------
        lat : float or array_like of floats
            Spherical (geocentric) latitude.
        radius : float or array_like of floats
            Radius, in metres.
        degrees : bool, optional
            If True, the input `lat` is given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = np.radians(lat)

        return self.omega**2 * radius * np.cos(lat) ** 2

    def lat_derivative(self, lat, radius, degrees=True):
        """Return latitude derivative, in 1/s**2.

        Parameters
        ----------
        lat : float or array_like of floats
            Spherical (geocentric) latitude.
        radius : float or array_like of floats
            Radius, in metres.
        degrees : bool, optional
            If True, the input `lat` is given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = np.radians(lat)

        return -self.omega**2 * radius**2 * np.cos(lat) * np.sin(lat)

    def gradient(self, lat, radius, degrees=True):
        """Return centrifugal force, in m/s**2.

        Parameters
        ----------
        lat : float or array_like of floats
            Spherical (geocentric) latitude.
        radius : float or array_like of floats
            Radius, in metres.
        degrees : bool, optional
            If True, the input `lat` is given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = np.radians(lat)

        cr = self.r_derivative(lat, radius, degrees=False)
        clat = 1 / r * self.lat_derivative(lat, radius, degrees=False)

        return np.sqrt(cr**2 + clat**2)


def _j2_to_flattening(j2, a, gm, omega):
    """Calculate flattening from J2, a, GM and omega.

    """
    _m1 = omega**2 * a**3 / gm

    def e2(e2, j2, _m1):
        """Compute e2 from J2.

        """
        e1 = np.sqrt(e2 / (1 - e2))
        q0 = 0.5 * ((1 + 3 / e1**2) * np.arctan(e1) - 3 / e1)
        return 3 * j2 + 2 / 15 * _m1 * np.sqrt(e2)**3 / q0 - e2

    _e2_0 = 3 * j2 + 2 / 15 * _m1
    _e2 = optimize.fsolve(e2, _e2_0, args=(j2, _m1), xtol=1e-10)[0]

    return 1 - np.sqrt(1 - _e2)


class LevelEllipsoid(Ellipsoid):
    """Class represents the gravity field of the level ellipsoid.

    This class intialize `Ellipsoid` class from
    `pygeoid.coordinates.ellipsoid`, so
    all geometrical methods and parameters are available too.

    Parameters
    ----------
    ellps : {'GRS80', 'WGS84', 'PZ90', 'GSK2011'}, optional
        Ellipsoid name. Default is 'GRS80'.
    """

    def __init__(self, ellps=None, **kwargs):
        if not kwargs:
            if ellps in LEVEL_ELLIPSOIDS:
                kwargs = LEVEL_ELLIPSOIDS[ellps]
            elif ellps is None or ellps.lower() == 'default':
                kwargs = LEVEL_ELLIPSOIDS[DEFAULT_LEVEL_ELLIPSOID]
            else:
                raise ValueError(
                    'No ellipsoid with name {:%s}, possible values \
                        are:\n{:%s}'.format(ellps,
                                            LEVEL_ELLIPSOIDS.keys()))

        if 'j2' in kwargs:
            kwargs['f'] = _j2_to_flattening(kwargs['j2'], kwargs['a'],
                                            kwargs['gm'], kwargs['omega'])
            self._j2 = kwargs['j2']

        self._gm = kwargs['gm']
        self._omega = kwargs['omega']

        super().__init__(self, **kwargs)

        # define useful short-named attributes
        self._m = self.omega**2 * self.a**2 * self.b / self.gm
        self._q0 = 0.5 * ((1 + 3 / self.e1**2) *
                          np.arctan(self.e1) - 3 / self.e1)

        if not hasattr(self, '_j2'):
            self._j2 = self.e2 / 3 * (1 - 2 / 15 * self.m * self.e1 / self._q0)

        self._q01 = 3 * (1 +
                         1 / self.e12) * (1 - np.arctan(self.e1) / self.e1) - 1

        self._surface_potential = self.gm / self.linear_eccentricity *\
            np.arctan(self.second_eccentricity) +\
            1 / 3 * self.omega ** 2 * self.a ** 2

        self._gamma_e = self.gm / (self.a *
                                   self.b) * (1 - self.m -
                                              self.m / 6 * self.e1 * self._q01 / self._q0)

        self._gamma_p = self.gm / self.a**2 *\
            (1 + self.m / 3 * self.e1 * self._q01 / self._q0)

        self._gravity_flattening = (self._gamma_p -
                                    self._gamma_e) / self._gamma_e

        self._k = (self.b * self._gamma_p -
                   self.a * self._gamma_e) / (self.a * self._gamma_e)

    @property
    def j2(self):
        """Return dynamic form factor J2.

        """
        return self._j2

    @property
    def gm(self):
        """Return geocentric gravitational constant.

        """
        return self._gm

    @property
    def omega(self):
        """Return angular velocity, in radians.

        """
        return self._omega

    @property
    def m(self):
        r"""Auxiliary constant.

        Notes
        -----
        .. math::
            m = \frac{{\omega}^2 a^2 b}{GM}.
        """
        return self._m

    #########################################################################
    # Potential
    #########################################################################
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

    #########################################################################
    # Normal gravity
    #########################################################################
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
    def mean_normal_gravity(self):
        """Return mean normal gravity over ellipsoid, in m/s**2.

        """
        return 4 * np.pi / self.surface_area * (self._gm -
                                                2 / 3 * self._omega**2 * self.a**2 * self.b)

    @property
    def gravity_flattening(self):
        """Return gravity flattening.

        f* = (gamma_p - gamma_e) / gamma_e
        """
        return self._gravity_flattening

    def conventional_gravity_coeffs(self):
        """Return coefficients for the conventional gravity formula.

        gamma_0 = gamma_e*(1 + beta*sin(lat)**2 - beta1*sin(2*lat)**2)
        """
        f4_coeff = -0.5 * self.f**2 + 2.5 * self.f * self.m
        return (self.gravity_flattening, 0.25 * f4_coeff)

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

    def height_correction(self, lat, height, degrees=True):
        """Return height correction, in m/s**2.

        Second-order approximation formula is used instead of -0.3086*height.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.
        height : float or array_like of floats
            Geodetic height, in metres.
        degrees : bool, optional
            If True, the input `lat` is given in degrees, otherwise radians.
            Default is True.
        """
        if degrees:
            lat = np.radians(lat)

        gammae = self.equatorial_normal_gravity
        out = -2 * gammae / self.a * (1 +
                                      self.f + self.m + (-3 * self.f +
                                                         2.5 * self.m) * np.sin(lat)**2) * height +\
            3 * gammae * height**2 / self.a**2

        return out

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

        u_deriv = self.gm / uE
        u_deriv += self.omega**2 * self.a ** 2 * E / uE * q1 / self._q0 * (
            0.5 * np.sin(rlat)**2 - 1 / 6)
        u_deriv -= self.omega**2 * u * np.cos(rlat)**2
        u_deriv *= -1 / w

        rlat_deriv = -self.omega**2 * self.a**2 / np.sqrt(uE) * _qr
        rlat_deriv += self.omega**2 * np.sqrt(uE)
        rlat_deriv *= -1 / w * np.sin(rlat) * np.cos(rlat)

        return np.sqrt(u_deriv**2 + rlat_deriv**2)

    #########################################################################
    # Spherical approximation
    #########################################################################
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

        out = 0
        for degree in range(1, n_max + 1):
            leg = special.eval_legendre(2 * degree, np.sin(lat))
            out += self.j2n(degree) * (self.a / radius) ** (2 * degree) * leg
        return self.gm / radius * (1 - out)

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
                                                             radius=radius,
                                                             degrees=False,
                                                             **kwargs)
        centrifugal = 0.5 * self.omega**2 * radius**2 * np.cos(lat)**2
        return gravitational_sph + centrifugal


NORMAL_GRAVITY_COEFFS = {
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

    if model is not None and model in NORMAL_GRAVITY_COEFFS:
        gamma_e, beta, beta1 = NORMAL_GRAVITY_COEFFS[model]
    else:
        msg = 'No formula with name {:%s}, possible values are:\n{:%s}'
        raise ValueError(msg.format(model, model.keys()))

    if degrees:
        lat = np.radians(lat)

    return gamma_e * (1 + beta * np.sin(lat)**2 - beta1 * np.sin(2 * lat)**2)
