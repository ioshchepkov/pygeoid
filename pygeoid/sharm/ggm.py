
import numpy as _np
import astropy.units as u


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
    coeffs : ~astropy.units.Quantity
        Dimensionless fully-normalized spherical harmonics coefficients
        with the sahpe (2, lmax+1, lmax+1).
        Where `lmax` is the maximum degree of the coefficients.
    gm : ~astropy.units.Quantity
        Gravitational parameter that is associated with the gravitational
        potential coefficients.
    r0 : ~astropy.units.Quantity
        Reference radius of the gravitational potential coefficients.
    coeffs : ~astropy.units.Quantity
        Uncertainties of the spherical harmonic coefficients. It should have
        the same shape as `coeffs`.
    ell : instance of the `pygeoid.reduction.ellipsoid.LevelEllipsoid`
        Reference ellipsoid to which noramal gravity field is referenced to.
        Default is `None` (default ellipsoid will be used).
    omega : ~astropy.units.Quantity
        Angular rotation rate of the body.

    References
    ----------
    .. [1] Barthelmes, Franz. ‘Definition of Functionals of the Geopotential
    and Their Calculation from Spherical Harmonic Models’. Deutsches
    GeoForschungsZentrum (GFZ), 2013. https://doi.org/10.2312/GFZ.b103-0902-26.
    """

    @u.quantity_input
    def __init__(self,
                 coeffs: u.dimensionless_unscaled,
                 gm: u.m**3 / u.s**2,
                 r0 : u.m,
                 errors: bool = None,
                 ell=None,
                 omega: 1 / u.s = None):

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
        return psi

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

    @u.quantity_input
    def gravitation(self, lat: u.deg, lon: u.deg, r: u.m, lmax=None) -> u.m / u.s**2:
        """Return gradient vector.

        The magnitude and the components of the gradient of the potential
        calculated on or above the ellipsoid without the centrifugal potential
        (eqs. 7 and 122 of STR09/02).

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Gravitation.
        """

        return self._gravitational.gradient(lat, lon, r, lmax=lmax)[-1]

    @u.quantity_input
    def gravity(self, lat: u.deg, lon: u.deg, r: u.m, lmax=None) -> u.m / u.s**2:
        """Return gravity value.

        The magnitude of the gradient of the potential calculated on or above
        the ellipsoid including the centrifugal potential (eqs. 7 and 121 − 124
        of STR09/02).

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Gravity.
        """
        return _np.squeeze(self._gravity.gradient(
            lat, lon, r, lmax=lmax)[-1])

    @u.quantity_input
    def gravity_disturbance(self,
                            lat: u.deg, lon: u.deg, r: u.m, lmax: int = None) -> u.mGal:
        """Return gravity disturbance.

        The gravity disturbance is defined as the magnitude of the gradient of
        the potential at a given point minus the magnitude of the gradient of
        the normal potential at the same point (eqs. 87 and 121 − 124 of STR09/02).

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Gravity disturbance.
        """
        rlat, _, u_ax = _transform.cartesian_to_ellipsoidal(
            *_transform.spherical_to_cartesian(lat, lon, r),
            self._ell)

        g = self._gravity.gradient(lat, lon, r, lmax)[-1]
        gamma = self._ell.normal_gravity(rlat, u_ax)

        return g - gamma

    @u.quantity_input
    def gravity_disturbance_sa(self,
                               lat: u.deg, lon: u.deg, r: u.m, lmax: int = None) -> u.mGal:
        """Return gravity disturbance in spherical approximation.

        The gravity disturbance calculated by spherical approximation (eqs. 92
        and 125 of STR09/02) on (h=0) or above (h>0) the ellipsoid.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Gravity disturbance.
        """

        return -self.anomalous_potential.r_derivative(lat, lon, r, lmax=lmax)

    @u.quantity_input
    def gravity_anomaly_sa(self,
                           lat: u.deg, lon: u.deg, r: u.m, lmax: int = None) -> u.mGal:
        """Return (Molodensky) gravity anomaly in spherical approximation.

        The gravity anomaly calculated by spherical approximation (eqs. 100 or
        104 and 126 of STR09/02). Unlike the classical gravity anomaly, the
        Molodensky gravity anomaly and the spherical approximation can be
        generalised to 3-d space, hence here it can be calculated on (h=0) or
        above (h>0) the ellipsoid.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Gravity anomaly.
        """

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

    @u.quantity_input
    def height_anomaly_ell(self, lat: u.deg, lon: u.deg, r: u.m,
                           ref_pot: u.m**2 / u.s**2 = None, lmax=None) -> u.m:
        """Return height anomaly above the ellispoid.

        The height anomaly can be generalised to a 3-d function, (sometimes
        called "generalised pseudo-height-anomaly"). Here it can be calculated
        on (h=0) or above (h>0) the ellipsoid, approximated by Bruns’ formula
        (eqs. 78 and 118 of STR09/02)

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        ref_pot : ~astropy.units.Quantity
            Reference potential value W0 for the zero degree term. Defaut is
            `None` (zero degree term is not considered).
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Anomaly height.
        """
        rlat, _, u_ax = _transform.cartesian_to_ellipsoidal(
            *_transform.spherical_to_cartesian(lat, lon, r), self._ell)

        T = self.anomalous_potential.potential(lat, lon, r, lmax=lmax)

        gamma = self._ell.normal_gravity(rlat, u_ax)

        zeta = _np.squeeze(T / gamma)

        if ref_pot is not None:
            zeta -= (ref_pot - self._ell.surface_potential) / gamma

        return zeta


class SHGravPotential:

    @u.quantity_input
    def __init__(self,
                 coeffs: u.dimensionless_unscaled,
                 gm: u.m**3 / u.s**2, r0 : u.m, omega: 1 / u.s = None,
                 errors: bool = None, lmax: int = None, copy: bool = False):

        self._coeffs = _SHCoeffs.from_array(coeffs, lmax=lmax, copy=copy)
        self.gm = gm
        self.r0 = r0
        self.omega = omega
        self.errors = errors

        if self.omega is not None:
            self.centrifugal = _Centrifugal(omega=self.omega)

    @u.quantity_input
    def potential(self,
                  lat: u.deg, lon: u.deg, r: u.m, lmax=None) -> u.m**2 / u.s**2:
        """Return potential value.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Potential.
        """

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)

        _, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon,
                                                               r, self.r0, lmax_comp)
        args = (_expand.in_coeff_potential, _expand.sum_potential,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r

        out = _np.squeeze(self.gm * ri * values)

        if self.omega is not None:
            out += self.centrifugal.potential(lat, r)

        return out

    @u.quantity_input
    def r_derivative(self,
                     lat: u.deg, lon: u.deg, r: u.m, lmax: int = None) -> u.m / u.s**2:
        """Return radial derivative of the potential.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Radial derivative.
        """

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        _, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                               self.r0, lmax_comp)

        args = (_expand.in_coeff_r_derivative, _expand.sum_potential,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = _np.squeeze(-self.gm * ri**2 * values)

        if self.omega is not None:
            out += self.centrifugal.r_derivative(lat, r)

        return out

    @u.quantity_input
    def lat_derivative(self,
                       lat: u.deg, lon: u.deg, r: u.m, lmax: int = None):
        """Return latitudinal derivative of the potential.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Latitudinal derivative.
        """

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        lat, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                                 self.r0, lmax_comp)

        args = (_expand.in_coeff_lat_derivative, _expand.sum_lat_derivative,
                lmax_comp, degrees, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = _np.squeeze(self.gm * ri * _np.cos(lat) * values)

        if self.omega is not None:
            out += self.centrifugal.lat_derivative(lat, r)

        return out

    @u.quantity_input
    def lon_derivative(self, lat: u.deg,
                       lon: u.deg, r: u.m, lmax: int = None):
        """Return longitudinal derivative of the potential.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).

        Returns
        -------
        ~astropy.units.Quantity
            Longitudinal derivative.
        """

        cilm, lmax_comp = _get_lmax(self._coeffs.coeffs, lmax=lmax)
        _, _, degrees, cosin, x, q = _expand.common_precompute(lat, lon, r,
                                                               self.r0, lmax_comp)
        m_coeff = _np.tile(degrees, (lmax_comp + 1, 1))

        args = (_expand.in_coeff_lon_derivative, _expand.sum_lon_derivative,
                lmax_comp, degrees, m_coeff, cosin, cilm)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / r
        out = -self.gm * ri * values

        return out.squeeze()

    @u.quantity_input
    def gradient(self,
                 lat: u.deg, lon: u.deg, r: u.m, lmax: int = None):
        """Return gradient vector.

        The magnitude and the components of the gradient of the potential
        calculated on or above the ellipsoid without the centrifugal potential
        (eqs. 7 and 122 of STR09/02).

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Spherical latitude.
        lon : ~astropy.units.Quantity
            Spherical longitude.
        r : ~astropy.units.Quantity
            Radial distance.
        lmax : int, optional
            Maximum degree of the coefficients. Default is `None` (use all
            the coefficients).
        """

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
            lat_d += self.centrifugal.lat_derivative(lat, r)
            rad_d += self.centrifugal.r_derivative(lat, r)

        # total
        clati = _np.atleast_2d(1 / _np.ma.masked_values(clat, 0.0))
        clati = clati.filled(0.0)

        total = _np.sqrt((ri * lat_d)**2 + (clati * ri * lon_d)**2 + rad_d**2)

        return (rad_d.squeeze(), lon_d.squeeze(),
                lat_d.squeeze(), total.squeeze())
