"""
Gravity field and geometry of the level ellipsoid.
"""

import numpy as np
import astropy.units as u
from scipy import (optimize, special)
from pygeoid.coordinates.ellipsoid import Ellipsoid


LEVEL_ELLIPSOIDS = {
    'GRS80': {'description': 'GRS 80',
              'a': 6378137.0 * u.m,
              'j2': 108263e-8 * u.dimensionless_unscaled,
              'gm': 3986005e8 * u.m**3 / u.s**2,
              'omega': 7292115e-11 / u.s},
    'WGS84': {'description': 'WGS 84',
              'a': 6378137.0 * u.m,
              'rf': 298.2572235630 * u.dimensionless_unscaled,
              'gm': 3986004.418e8 * u.m**3 / u.s**2,
              'omega': 7292115e-11 / u.s},
    'PZ90': {'description': 'PZ 90.11',
             'a': 6378136.0 * u.m,
             'rf': 298.25784 * u.dimensionless_unscaled,
             'gm': 3986004.418e8 * u.m**3 / u.s**2,
             'omega': 7292115e-11 * u.rad / u.s},
    'GSK2011': {'description': 'GSK-2011',
                'a': 6378136.5 * u.m,
                'rf': 298.2564151 * u.dimensionless_unscaled,
                'gm': 3986004.415e8 * u.m**3 / u.s**2,
                'omega': 7292115e-11 / u.s},
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

    @u.quantity_input
    def __init__(self, omega=7292115e-11 * u.rad / u.s):
        self.omega = omega

    @u.quantity_input
    def potential(self, lat: u.deg, radius: u.m) -> u.m**2 / u.s**2:
        """Return centrifugal potential.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical (geocentric) latitude.
        radius : ~astropy.units.Quantity
            Radius.
        """
        return 0.5 * self.omega**2 * radius**2 * np.cos(lat)**2

    @u.quantity_input
    def r_derivative(self, lat: u.deg, radius: u.m):
        """Return radial derivative.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical (geocentric) latitude.
        radius : ~astropy.units.Quantity
            Radius.
        """
        return self.omega**2 * radius * np.cos(lat) ** 2

    @u.quantity_input
    def lat_derivative(self, lat: u.deg, radius: u.m):
        """Return latitude derivative.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical (geocentric) latitude.
        radius : ~astropy.units.Quantity
            Radius.
        """
        return -self.omega**2 * radius**2 * np.cos(lat) * np.sin(lat)

    @u.quantity_input
    def gradient(self, lat: u.deg, radius: u.deg) -> u.m / u.s**2:
        """Return centrifugal force.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical (geocentric) latitude.
        radius : ~astropy.units.Quantity
            Radius.
        """
        cr = self.r_derivative(lat, radius)
        clat = 1 / radius * self.lat_derivative(lat, radius)

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
            kwargs['f'] = _j2_to_flattening(
                kwargs['j2'].si.value,
                kwargs['a'].si.value,
                kwargs['gm'].si.value,
                kwargs['omega'].si.value) * u.dimensionless_unscaled

            self._j2 = kwargs['j2']

        self._gm = kwargs['gm']
        self._omega = kwargs['omega']

        kwargs_nounits = {key: x.si.value for key, x in kwargs.items() if hasattr(x, 'unit')}

        super().__init__(self, **kwargs_nounits)

        # define useful short-named attributes
        self._m = self.omega**2 * self.a**2 * self.b / self.gm
        self._q0 = 0.5 * ((1 + 3 / self.e1**2) *
                          np.arctan(self.e1).value * u.dimensionless_unscaled - 3 / self.e1)

        if not hasattr(self, '_j2'):
            self._j2 = self.e2 / 3 * (1 - 2 / 15 * self.m * self.e1 / self._q0)

        self._q01 = 3 * (1 +
                         1 / self.e12) * (1 - np.arctan(self.e1).value * u.dimensionless_unscaled / self.e1) - 1

        self._surface_potential = self.gm / self.linear_eccentricity *\
            np.arctan(self.second_eccentricity).value * u.dimensionless_unscaled +\
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
        """Return normal gravity potential on the ellipsoid.

        Value of the normal gravity potential on the ellipsoid, or on the
        equipotential surface U(x, y, z) = U_0.

        """
        return self._surface_potential

    @u.quantity_input
    def _q(self, u_ax: u.m):
        """Return auxiliary function q(u).

        """
        E = self.linear_eccentricity
        return 0.5 * ((1 + 3 * u_ax**2 / E**2) * np.arctan2(E, u_ax).value *
                      u.dimensionless_unscaled - 3 * u_ax / E)

    @u.quantity_input
    def gravitational_potential(self, rlat: u.deg, u_ax: u.m) -> u.m**2 / u.s**2:
        """Return normal gravitational potential V.

        Calculate normal gravitational potential from the rigorous formula.

        Parameters
        ----------
        rlat : ~astropy.units.Quantity
            Reduced latitude.
        u_ax : ~astropy.units.Quantity
            Polar axis of the ellipsoid passing through the point.

        Returns
        -------
        ~astropy.units.Quantity
            Normal gravitational potential.
        """
        E = self.linear_eccentricity
        arctanEu = np.arctan2(E, u_ax).value * u.dimensionless_unscaled
        _qr = self._q(u_ax) / self._q0

        return (self.gm / E) * arctanEu + \
            0.5 * self.omega**2 * self.a**2 * \
            _qr * (np.sin(rlat)**2 - 1 / 3)

    @u.quantity_input
    def gravity_potential(self, rlat: u.deg, u_ax: u.m) -> u.m**2 / u.s**2:
        """Return normal gravity potential U.

        Calculate normal gravity potential from the rigorous formula.

        Parameters
        ----------
        rlat : ~astropy.units.Quantity
            Reduced latitude.
        u_ax : ~astropy.units.Quantity
            Polar axis of the ellipsoid passing through the point.

        Returns
        -------
        ~astropy.units.Quantity
            Normal gravity potential.
        """
        gravitational = self.gravitational_potential(rlat, u_ax)
        centrifugal = 0.5 * self.omega**2 * (u_ax**2 +
                                             self.linear_eccentricity**2) * np.cos(rlat)**2
        return gravitational + centrifugal

    #########################################################################
    # Normal gravity
    #########################################################################
    @property
    def equatorial_normal_gravity(self):
        """Return normal gravity at the equator.

        """
        return self._gamma_e

    @property
    def polar_normal_gravity(self):
        """Return normal gravity at the poles.

        """
        return self._gamma_p

    @property
    def mean_normal_gravity(self):
        """Return mean normal gravity over ellipsoid.

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

    @u.quantity_input
    def surface_normal_gravity(self, lat: u.deg) -> u.m / u.s**2:
        """Return normal gravity on the ellipsoid.

        Calculate normal gravity value on the level ellipsoid by the rigorous
        formula of Somigliana.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
            Normal gravity on the ellipsoid.
        """
        return self._gamma_e * (1 + self._k * np.sin(lat) ** 2) / self._w(lat)

    @u.quantity_input
    def surface_vertical_normal_gravity_gradient(self, lat: u.deg):
        """Return the vertical gravity gradient on the ellipsoid.

        Vertical gradient of the normal gravity at the reference ellipsoid.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.
        """
        gamma = self.surface_normal_gravity(lat)
        return -2 * gamma * self.average_curvature(lat) - 2 * self.omega ** 2

    @u.quantity_input
    def height_correction(self, lat: u.deg, height: u.m) -> u.m / u.s**2:
        """Return height correction.

        Second-order approximation formula is used instead of -0.3086*height.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.
        height : ~astropy.units.Quantity
            Geodetic height.
        """
        gammae = self.equatorial_normal_gravity
        out = -2 * gammae / self.a * (1 +
                                      self.f + self.m + (-3 * self.f +
                                                         2.5 * self.m) * np.sin(lat)**2) * height +\
            3 * gammae * height**2 / self.a**2

        return out

    @u.quantity_input
    def normal_gravity(self, rlat: u.deg, u_ax: u.m) -> u.m / u.s**2:
        """Return normal gravity.

        Calculate normal gravity value at any arbitrary point by the rigorous
        closed formula.

        Parameters
        ----------
        rlat : ~astropy.units.Quantity
            Reduced latitude.
        u_ax : ~astropy.units.Quantity
            Polar axis of the ellipsoid passing through the point.

        Returns
        -------
        ~astropy.units.Quantity
            Normal gravity.
        """

        E = self.linear_eccentricity
        _qr = self._q(u_ax) / self._q0

        uE = u_ax**2 + E**2
        w = np.sqrt((u_ax**2 + E**2 * np.sin(rlat)**2) / uE)
        arctan2Eu = np.arctan2(E, u_ax).value * u.dimensionless_unscaled
        q1 = 3 * (1 + u_ax**2 / E**2) * (1 - u_ax / E * arctan2Eu) - 1

        u_deriv = self.gm / uE
        u_deriv += self.omega**2 * self.a ** 2 * E / uE * q1 / self._q0 * (
            0.5 * np.sin(rlat)**2 - 1 / 6)
        u_deriv -= self.omega**2 * u_ax * np.cos(rlat)**2
        u_deriv *= -1 / w

        rlat_deriv = -self.omega**2 * self.a**2 / np.sqrt(uE) * _qr
        rlat_deriv += self.omega**2 * np.sqrt(uE)
        rlat_deriv *= -1 / w * np.sin(rlat) * np.cos(rlat)

        return np.sqrt(u_deriv**2 + rlat_deriv**2)

    #########################################################################
    # Spherical approximation
    #########################################################################
    @u.quantity_input
    def j2n(self, n: int) -> u.dimensionless_unscaled:
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

    @u.quantity_input
    def gravitational_potential_sph(
            self, lat: u.deg, radius: u.m, n_max: int = 4) -> u.m**2 / u.s**2:
        """Return normal gravitational potential V.

        Calculate normal gravitational potential from spherical approximation.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical (geocentric) latitude.
        radius : ~astropy.units.Quantity
            Radius, in metres.
        n_max : int
            Maximum degree.
        """
        out = 0
        for degree in range(1, n_max + 1):
            leg = special.eval_legendre(2 * degree, np.sin(lat).value)
            out += self.j2n(degree) * (self.a / radius) ** (2 * degree) * leg
        return self.gm / radius * (1 - out)

    @u.quantity_input
    def gravity_potential_sph(self,
                              lat: u.deg, radius: u.m, n_max: int = 4) -> u.m**2 / u.s**2:
        """Return normal gravitational potential V.

        Calculate normal gravitational potential from spherical approximation.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical (geocentric) latitude.
        radius : ~astropy.units.Quantity
            Radius, in metres.
        n_max : int
            Maximum degree.
        """
        gravitational_sph = self.gravitational_potential_sph(lat=lat,
                                                             radius=radius,
                                                             n_max=n_max)
        centrifugal = 0.5 * self.omega**2 * radius**2 * np.cos(lat)**2
        return gravitational_sph + centrifugal


NORMAL_GRAVITY_COEFFS = {
    'helmert': (978030 * u.mGal,
                0.005302 * u.dimensionless_unscaled,
                0.000007 * u.dimensionless_unscaled),
    'helmert_14mGal': (
        (978030 - 14) * u.mGal,
        0.005302 * u.dimensionless_unscaled,
        0.000007 * u.dimensionless_unscaled),
    '1930': (978049 * u.mGal,
             0.0052884 * u.dimensionless_unscaled,
             0.0000059 * u.dimensionless_unscaled),
    '1930_14mGal': (
        (978049 - 14) * u.mGal,
        0.0052884 * u.dimensionless_unscaled,
        0.0000059 * u.dimensionless_unscaled),
    '1967': (978031.8 * u.mGal,
             0.0053024 * u.dimensionless_unscaled,
             0.0000059 * u.dimensionless_unscaled),
    '1980': (978032.7 * u.mGal,
             0.0053024 * u.dimensionless_unscaled,
             0.0000058 * u.dimensionless_unscaled)}


@u.quantity_input
def surface_normal_gravity_clairaut(lat: u.deg, model: str = None) -> u.m / u.s**2:
    """Return normal gravity from the first Clairaut formula.

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    model : {'helmert', 'helmert_14mGal', '1930',\
            '1930_14mGal', '1967', '1980'}
        Which gravity formula will be used.
    """

    if model is not None and model in NORMAL_GRAVITY_COEFFS:
        gamma_e, beta, beta1 = NORMAL_GRAVITY_COEFFS[model]
    else:
        msg = 'No formula with name {:%s}, possible values are:\n{:%s}'
        raise ValueError(msg.format(model, model.keys()))

    return gamma_e * (1 + beta * np.sin(lat)**2 - beta1 * np.sin(2 * lat)**2)
