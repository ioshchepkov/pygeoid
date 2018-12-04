
import numpy as _np

from pyshtools.shclasses import SHGravCoeffs as _SHGravCoeffs
from pyshtools.shclasses import SHCoeffs as _SHCoeffs

from pygeoid.reduction.normal import Centrifugal as _Centrifugal
from pygeoid.reduction.normal import LevelEllipsoid as _LevelEllipsoid
from pygeoid.coordinates import transform as _transform
from pygeoid.sharm import expand as _expand
from pygeoid.sharm.utils import get_lmax as _get_lmax


class GlobalGravityFieldModel:
    """
    Class for working with the global gravity field models.

    The theory and formulas used by this class as in the ICGEM calculation
    service and described in the Scientific Technical Report STR09/02 [1]_.

    Parameters
    ----------
    coeffs  : array_like
        Spherical harmonics coefficients with the sahpe (2, lmax+1, lmax+1).
        Where `lmax` is the maximum degree of the coefficients.
    gm  : float
        Gravitational parameter that is associated with the gravitational
        potential coefficients.
    r0  : float
        Reference radius of the gravitational potential coefficients, in
        metres.
    errors  : array_like, optional
        Uncertainties of the spherical harmonic coefficients. It should have
        the same shape as `coeffs`.
    ell : instance of the `pygeoid.reduction.ellipsoid.LevelEllipsoid`
        Reference ellipsoid to which noramal gravity field is referenced to.
        Default is `None` (default ellipsoid will be used).
    omega : float, optional
        Angular rotation rate of the body, in rad/s.

    References
    ----------
    .. [1] Barthelmes, Franz. ‘Definition of Functionals of the Geopotential
    and Their Calculation from Spherical Harmonic Models’. Deutsches
    GeoForschungsZentrum (GFZ), 2013. https://doi.org/10.2312/GFZ.b103-0902-26.
    """

    def __init__(self, coeffs, gm, r0, errors=None, ell=None, omega=None):

        if ell is not None:
            self._ell = ell
        else:
            self._ell = _LevelEllipsoid()

        if omega is None:
            omega = self._ell.omega

        self._coeffs = _SHGravCoeffs.from_array(coeffs=coeffs, gm=gm, r0=r0,
                                                errors=errors, omega=omega, copy=True)

    @property
    def resolution(self):
        """Return half-wavelength of the model.

        """
        psi = 4 * _np.arcsin(1 / (self._coeffs.lmax + 1))
        return _np.degrees(psi)

    @property
    def _gravitational(self):
        """Return `SHGravPotential` class instance for the gravitational potential.

        """
        return SHGravPotential(coeffs=self._coeffs.coeffs, gm=self._coeffs.gm,
                               r0=self._coeffs.r0, omega=None, copy=False)

    @property
    def gravitational_potential(self):
        """Return `SHGravPotential` class instance for the gravitational potential.

        """
        return self._gravitational

    @property
    def _gravity(self):
        """Return `SHGravPotential` class instance for the gravity potential.

        """
        return SHGravPotential(coeffs=self._coeffs.coeffs, gm=self._coeffs.gm,
                               r0=self._coeffs.r0, omega=self._coeffs.omega)

    @property
    def gravity_potential(self):
        """Return `SHGravPotential` class instance for the gravity potential.

        """
        return self._gravity

    @property
    def _anomalous(self):
        """Return `SHGravPotential` class instance for anomalous potential.

        """
        _dc = _np.zeros_like(self._coeffs.coeffs)
        _dc[0, 0, 0] = 1 + (self._ell.gm - self._coeffs.gm) / self._coeffs.gm

        ngm = self._ell.gm / self._coeffs.gm
        nr = self._ell.a / self._coeffs.r0

        zmax = 5
        for i in range(1, zmax + 1):
            _dc[0, 2 * i, 0] += (ngm * _np.power(nr, 2 * i) *
                                 -self._ell.j2n(i) / _np.sqrt(4 * i + 1))

        return SHGravPotential(self._coeffs.coeffs - _dc,
                               gm=self._coeffs.gm, r0=self._coeffs.r0,
                               omega=None)

    @property
    def anomalous_potential(self):
        """Return `SHGravPotential` class instance for anomalous potential.

        """
        return self._anomalous

    @property
    def normal_potential(self):
        """Return normal potential class instance.

        """
        return self._ell

    def gravitation(self, lat, lon, r, lmax=None, degrees=True):
        """Return gradient vector.

        The magnitude and the components of the gradient of the potential
        calculated on or above the ellipsoid without the centrifugal potential
        (eqs. 7 and 122 of STR09/02).

        Parameters
        ----------
        lat : float
            Latitude, in degrees
        lon : float
            Longitude, in degrees
        r   : float
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float or array
            Gravitation, in m/s**2.
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        return self._gravitational.gradient(lat, lon, r, lmax=lmax,
                                            degrees=False)[-1]

    def gravity(self, lat, lon, r, lmax=None, degrees=True):
        """Return gravity value.

        The magnitude of the gradient of the potential calculated on or above
        the ellipsoid including the centrifugal potential (eqs. 7 and 121 − 124
        of STR09/02).

        Parameters
        ----------
        lat : float
            Latitude, in degrees
        lon : float
            Longitude, in degrees
        r   : float
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float
            Gravity, in m/s**2.
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        return _np.squeeze(self._gravity.gradient(lat, lon, r, lmax=lmax,
                                                  degrees=False)[-1])

    def gravity_disturbance(self, lat, lon, r, lmax=None, degrees=True):
        """Return gravity disturbance.

        The gravity disturbance is defined as the magnitude of the gradient of
        the potential at a given point minus the magnitude of the gradient of
        the normal potential at the same point (eqs. 87 and 121 − 124 of STR09/02).

        Parameters
        ----------
        lat : float
            Latitude, in degrees
        lon : float
            Longitude, in degrees
        r   : float
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float
            Gravity disturbance, in m/s**2
        """
        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        rlat, _, u = _transform.cartesian_to_ellipsoidal(
            *_transform.spherical_to_cartesian(lat, lon, r, degrees=False),
            self._ell, degrees=False)

        g = self._gravity.gradient(lat, lon, r, lmax, degrees=False)[-1]
        gamma = self._ell.normal_gravity(rlat, u, degrees=False)

        return g - gamma

    def gravity_disturbance_sa(self, lat, lon, r, lmax=None, degrees=True):
        """Return gravity disturbance in spherical approximation.

        The gravity disturbance calculated by spherical approximation (eqs. 92
        and 125 of STR09/02) on (h=0) or above (h>0) the ellipsoid.

        Parameters
        ----------
        lat : float
            Latitude, in degrees
        lon : float
            Longitude, in degrees
        r   : float
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float
            Gravity disturbance, in m/s**2
        """

        return -self.anomalous_potential.r_derivative(lat, lon, r, lmax=lmax,
                                                      degrees=degrees)

    def gravity_anomaly_sa(self, lat, lon, r, lmax=None, degrees=True):
        """Return (Molodensky) gravity anomaly in spherical approximation.

        The gravity anomaly calculated by spherical approximation (eqs. 100 or
        104 and 126 of STR09/02). Unlike the classical gravity anomaly, the
        Molodensky gravity anomaly and the spherical approximation can be
        generalised to 3-d space, hence here it can be calculated on (h=0) or
        above (h>0) the ellipsoid.

        Parameters
        ----------
        lat : float
            Latitude, in degrees
        lon : float
            Longitude, in degrees
        r   : float
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float
            Gravity anomaly, in m/s**2
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        coeffs = self._anomalous._coeffs.coeffs
        cilm, lmax_comp = _get_lmax(coeffs, lmax=lmax)
        _, _, degrees, cosin, x, q = _expand.common_precompute(
            lat, lon, r, self._coeffs.r0, lmax_comp)

        args = (_expand.in_coeff_gravity_anomaly, _expand.sum_potential,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = self._coeffs.gm * ri**2 * values

        return _np.squeeze(out)

    def height_anomaly_ell(self, lat, lon, r, ref_pot=None, lmax=None, degrees=True):
        """Return height anomaly above the ellispoid.

        The height anomaly can be generalised to a 3-d function, (sometimes
        called "generalised pseudo-height-anomaly"). Here it can be calculated
        on (h=0) or above (h>0) the ellipsoid, approximated by Bruns’ formula
        (eqs. 78 and 118 of STR09/02)

        Parameters
        ----------
        lat : float
            Latitude, in degrees
        lon : float
            Longitude, in degrees
        r   : float
            Radial distance, im meters
        ref_pot : float, in m**2 / s**2
            Reference potential value W0 for the zero degree term. Defaut is
            `None` (zero degree term is not considered).
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float
            Anomaly height, in meters
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        rlat, _, u = _transform.cartesian_to_ellipsoidal(
            *_transform.spherical_to_cartesian(lat, lon, r, degrees=False),
            self._ell, degrees=False)

        T = self.anomalous_potential.potential(lat, lon, r, lmax=lmax,
                                               degrees=False)

        gamma = self._ell.normal_gravity(rlat, u, degrees=False)

        zeta = _np.squeeze(T / gamma)

        if ref_pot is not None:
            zeta -= (ref_pot - self._ell.surface_potential) / gamma

        return zeta


class SHGravPotential:

    def __init__(self, coeffs, gm, r0, omega=None, errors=None, lmax=None,
                 copy=False):

        #self._coeffs = _SHGravCoeffs.from_array(coeffs, gm=gm, r0=r0,
        #                                        omega=omega, errors=errors, lmax=lmax, copy=copy)
        self._coeffs = _SHCoeffs.from_array(coeffs, lmax=lmax, copy=copy)
        self.gm = gm
        self.r0 = r0
        self.omega = omega
        self.errors = errors

        if self.omega is not None:
            self.centrifugal = _Centrifugal(omega=self.omega)

    def potential(self, lat, lon, r, lmax=None, degrees=True):
        """Return potential value.

        Parameters
        ----------
        lat : float or array
            Latitude, in degrees
        lon : float or array
            Longitude, in degrees
        r   : float or array
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float or array
            Potential, in m**2/s**2
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)

        _, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon,
                                                               r, self.r0, lmax_comp)
        args = (_expand.in_coeff_potential, _expand.sum_potential,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r

        out = _np.squeeze(self.gm * ri * values)

        if self.omega is not None:
            out += self.centrifugal.potential(lat, r, degrees=False)

        return out

    def r_derivative(self, lat, lon, r, lmax=None, degrees=True):
        """Return radial derivative of the potential, in m/s**2.

        Parameters
        ----------
        lat : float or array
            Latitude, in degrees
        lon : float or array
            Longitude, in degrees
        r   : float or array
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float or array
            Radial derivative, in m/s**2
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        _, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                               self.r0, lmax_comp)

        args = (_expand.in_coeff_r_derivative, _expand.sum_potential,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = _np.squeeze(-self.gm * ri**2 * values)

        if self.omega is not None:
            out += self.centrifugal.r_derivative(lat, r, degrees=False)

        return out

    def lat_derivative(self, lat, lon, r, lmax=None, degrees=True):
        """Return latitudinal derivative of the potential, in m/s**2.

        Parameters
        ----------
        lat : float or array
            Latitude, in degrees
        lon : float or array
            Longitude, in degrees
        r   : float or array
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float or array
            Latitudinal derivative, in m/s**2
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        lat, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                                 self.r0, lmax_comp)

        args = (_expand.in_coeff_lat_derivative, _expand.sum_lat_derivative,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = _np.squeeze(self.gm * ri * _np.cos(lat) * values)

        if self.omega is not None:
            out += self.centrifugal.lat_derivative(lat, r, degrees=False)

        return out

    def lon_derivative(self, lat, lon, r, lmax=None, degrees=True):
        """Return longitudinal derivative of the potential, in m/s**2.

        Parameters
        ----------
        lat : float or array
            Latitude, in degrees
        lon : float or array
            Longitude, in degrees
        r   : float or array
            Radial distance, im meters
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.

        Returns
        -------
        float or array
            Longitudinal derivative, in m/s**2
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        _, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                               self.r0, lmax_comp)
        m_coeff = _np.tile(degrees, (lmax_comp + 1, 1))

        args = (_expand.in_coeff_lon_derivative, _expand.sum_lon_derivative,
                lmax_comp, degrees, m_coeff, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = -self.gm * ri * values

        return _np.squeeze(out)

    def gradient(self, lat, lon, r, lmax=None, degrees=True):
        """Return gradient vector.

        The magnitude and the components of the gradient of the potential
        calculated on or above the ellipsoid without the centrifugal potential
        (eqs. 7 and 122 of STR09/02).

        Parameters
        ----------
        lat : float
            Latitude, in degrees.
        lon : float
            Longitude, in degrees.
        r   : float
            Radial distance, im meters.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.
        """

        if degrees:
            lat = _np.radians(lat)
            lon = _np.radians(lon)

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        lat, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                                 self.r0, lmax_comp)

        m_coeff = _np.tile(degrees, (lmax_comp + 1, 1))
        args = (_expand.in_coeff_gradient, _expand.sum_gradient, lmax_comp, degrees,
                m_coeff, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        gmri = self.gm * ri
        clat = _np.cos(lat)

        lat_d = gmri * clat * values[:, :, 0]
        lon_d = -gmri * values[:, :, 1]
        rad_d = -self.gm * ri**2 * values[:, :, 2]

        if self.omega is not None:
            lat_d += self.centrifugal.lat_derivative(lat, r, degrees=False)
            rad_d += self.centrifugal.r_derivative(lat, r, degrees=False)

        # total
        clati = _np.atleast_2d(1 / _np.ma.masked_values(clat, 0.0))
        clati = clati.filled(0.0)

        total = _np.sqrt((ri * lat_d)**2 + (clati * ri * lon_d)**2 + rad_d**2)

        return _np.squeeze([rad_d, lon_d, lat_d, total])
