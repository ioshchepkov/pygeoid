
import numpy as _np
import astropy.units as u


from pyshtools.shclasses import SHGravCoeffs as _SHGravCoeffs
from pyshtools.shclasses import SHCoeffs as _SHCoeffs


from pygeoid.potential.core import PotentialBase as _PotentialBase
from pygeoid.potential.core import CompositePotential as _CompositePotential
from pygeoid.potential.centrifugal import Centrifugal as _Centrifugal
from pygeoid.potential.normal import LevelEllipsoid as _LevelEllipsoid
from pygeoid.coordinates import transform as _transform
from pygeoid.constants.iers2010 import k20
from pygeoid.sharm import expand as _expand
from pygeoid.sharm.utils import get_lmax as _get_lmax


class GlobalGravityFieldModel:
    """Class for working with the global gravity field models.

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
    tide_system : {'zero-tide', 'non-tidal'},
        Tide system of the model. Can be 'zero-tide', 'non-tidal' or None.
        Default is None, i.e. model's system is used. If the tide system
        is not defined, it will be impossible to convert between them.
    lmax : int, optional
        Maximum degree of the coefficients. Default is `None` (use all
        the coefficients).
    errors : ~astropy.units.Quantity
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
                 tide_system: str = None,
                 lmax: int = None,
                 errors: bool = None,
                 ell=None,
                 omega: 1 / u.s = None):

        if tide_system in ('zero-tide', 'non-tidal', None):
            self._tide_system = tide_system
        else:
            raise ValueError('Not a valid `tide_system`!')

        if ell is not None:
            self._ell = ell
        else:
            self._ell = _LevelEllipsoid()

        if omega is None:
            omega = self._ell.omega

        coeffs = coeffs.astype(_np.float128)
        self._coeffs = _SHGravCoeffs.from_array(coeffs=coeffs, gm=gm, r0=r0,
                                                lmax=lmax, errors=errors, omega=omega, copy=True)

    def to_tide_system(self, tide_system: str):
        r"""Convert model to another tide system.

        Parameters
        ----------
        tide_system : {'zero-tide', 'non-tidal'}
            Tide system to convert the model.
            Can be 'zero-tide' or 'non-tidal'.

        Notes
        -----
        Some GGMs provide the coefficient :math:`C_{20}` in the tide-free (non-tidal)
        system (:math:`C_{20}^{NT}`); other GGMs provide it in the zero-tide
        system (:math:`C_{20}^{ZT}`). Their relationship is given by (see [2]_)

        .. math::
            C_{20}^{ZT} = C_{20}^{NT} + k_{20} \frac{r_0}{GM} A' \left(\frac{r_0}{a}\right)^2

        where :math:`k_{20}` is the conventional Love number (:math:`k_{20} = 0.30190`,
        see Petit and Luzum 2010), :math:`GM` is the geocentric gravitational constant,
        and :math:`r0` is the distance scaling factor used in
        the generation of the tide-free GGM. The parameters :math:`A’` and
        :math:`a` are given in [2]_.

        References
        ----------
        .. [2] Sánchez, L. et al. Strategy for the realisation of the
            International Height Reference System (IHRS). J Geod 95, 33 (2021).

        """

        if tide_system not in ('zero-tide', 'non-tidal'):
            raise ValueError('Not a valid `tide_system`!')
        elif self._tide_system == tide_system:
            raise ValueError('Nothing to do, the model is already in the '
                             'desired tide system!')
        elif self._tide_system is None:
            raise ValueError('Tide system of the model is None (not specified)!')

        A = -1.9444 * u.m**2 / u.s**2
        a = _LevelEllipsoid('GRS80').a

        r0a = self._coeffs.r0 / a
        r0gm = self._coeffs.r0 / self._coeffs.gm
        corr = k20 * r0gm * A * r0a**2

        if self._tide_system == 'zero-tide' and tide_system == 'non-tidal':
            dc20 = -corr
            self._tide_system = 'non-tidal'
        elif self._tide_system == 'non-tidal' and tide_system == 'zero-tide':
            dc20 = corr
            self._tide_system = 'zero-tide'

        c20 = self._coeffs.coeffs[0][2][0] + dc20
        self._coeffs.set_coeffs(c20, 2, 0)

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

    @u.quantity_input
    def gravitational_potential(self, position) -> u.m**2 / u.s**2:
        """Return gravitational potential.

        This is the potential of the gravitational field (without the
        centrifugal potential) at the given point (h,λ,φ).
        (eqs. 1 and 108 of STR09/02)

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravitational potential.

        """
        return self._gravitational.potential(position)

    @property
    def _gravity(self):
        """Return `SHGravPotential` class instance for the gravity potential.

        """
        # return SHGravPotential(coeffs=self._coeffs.coeffs, gm=self._coeffs.gm,
        #                       r0=self._coeffs.r0, omega=self._coeffs.omega)
        centrifugal = _Centrifugal(omega=self._coeffs.omega)

        return _CompositePotential(
            gravitational=self._gravitational,
            centrifugal=centrifugal)

    @u.quantity_input
    def gravity_potential(self, position) -> u.m**2 / u.s**2:
        """Return gravity potential.

        This is the potential of the gravity field of the Earth (including the
        centrifugal potential) at the given point (h,λ,φ).
        (eqs. 5 and 108 + 123 of STR09/02)

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravity potential.

        """
        return self._gravity.potential(position)

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

    @u.quantity_input
    def anomalous_potential(self, position) -> u.m**2 / u.s**2:
        """Return anomalous potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Anomalous potential.

        """
        return self._anomalous.potential(position)

    @u.quantity_input
    def gravitation(self, position) -> u.m / u.s**2:
        """Return gradient vector.

        Gravitation is the magnitude of the gravitational vector (gradient of
        the attraction potential without centrifugal potential) at the given
        point (h,λ,φ) (eq. 122 of STR09/02).

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravitation.
        """
        return self._gravitational.gradient(position)

    @u.quantity_input
    def gravity(self, position) -> u.m / u.s**2:
        """Return gravity value.

        Gravity is the magnitude of the gravity vector (gradient of the gravity
        potential which includes the centrifugal potential) at the given point
        (h,λ,φ) (eqs. 7 and 121 − 124 of STR09/02).

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravity.
        """
        return _np.squeeze(self._gravity.gradient(position, 'spherical'))

    @u.quantity_input
    def gravity_disturbance(self, position) -> u.mGal:
        """Return gravity disturbance.

        The gravity disturbance is defined as the magnitude of the gradient of
        the gravity potential at a given point minus the magnitude
        of the gradient of the normal potential at the same point
        (eqs. 29, 87 and 121 − 124 of STR09/02).

        If you pass position based on the physical height instead of geodetic
        height, then the gravity anomaly will be returned, not gravity
        disturbance.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravity disturbance.
        """
        g = self._gravity.gradient(position)

        ellharm = position.represent_as('ellipsoidalharmonic')
        gamma = self._ell.normal_gravity(ellharm.rlat, ellharm.u_ax)

        return g - gamma

    @u.quantity_input
    def gravity_disturbance_sa(self, position) -> u.mGal:
        """Return gravity disturbance in spherical approximation.

        Spherical approximation of the gravity disturbance means that the real
        gradient (direction of the plumbline) is replaced by it’s radial
        component.

        (eqs. 92 and 125 of STR09/02)

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravity disturbance in spherical approximation.
        """

        return -self._anomalous.r_derivative(position)

    @u.quantity_input
    def gravity_anomaly_sa(self, position) -> u.mGal:
        """Return (Molodensky) gravity anomaly in spherical approximation.

        The gravity anomaly calculated by spherical approximation (eqs. 100 or
        104 and 126 of STR09/02).

        Spherical approximation of the gravity anomaly means that the real
        gradient (direction of the plumbline) is replaced by it’s radial
        component.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Gravity anomaly.
        """
        sph = position.represent_as('spherical')

        coeffs = self._anomalous._coeffs.coeffs
        lmax = self._anomalous._coeffs.lmax
        _, _, degrees, cosin, x, q = _expand.common_precompute(
            sph.lat, sph.lon, sph.distance, self._coeffs.r0, lmax)

        args = (_expand.in_coeff_gravity_anomaly, _expand.sum_potential,
                lmax, degrees, cosin, coeffs)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / sph.distance
        out = self._coeffs.gm * ri**2 * values

        return _np.squeeze(out)

    @u.quantity_input
    def height_anomaly_ell(self, position,
                           ref_pot: u.m**2 / u.s**2 = None) -> u.m:
        """Return height anomaly above the ellipsoid.

        The so called "height anomaly" is an approximation of the geoid
        according to Molodensky's theory. It is equal to the geoid over sea.
        Here the generalised height anomaly at the given point (h,λ,φ)
        is approximated by Bruns’ formula:
        disturbance_potential(h,λ,φ) / normal_gravity(h,φ)
        (eq. 78 and 118 of STR09/02)

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.
        ref_pot : ~astropy.units.Quantity
            Reference potential value W0 for the zero degree term. Defaut is
            `None` (zero degree term is not considered).

        Returns
        -------
        ~astropy.units.Quantity
            Height anomaly.
        """

        apot = self._anomalous.potential(position)

        ellharm = position.represent_as('ellipsoidalharmonic')
        gamma = self._ell.normal_gravity(ellharm.rlat, ellharm.u_ax)

        zeta = _np.squeeze(apot / gamma)

        if ref_pot is not None:
            zeta -= (ref_pot - self._ell.surface_potential) / gamma

        return zeta

    @u.quantity_input
    def vertical_deflection_abs(self, position) -> u.arcsec:
        """Return gravimetric deflection of the vertical (DoV).

        This is the magnitude of the gravimetric deflection of the vertical.
        It is the angle between the vector of gravity and the vector of normal gravity
        both at the same point (h,λ,φ).

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Deflection of the vertical.

        Notes
        -----
        The gravimetric vertical deflection computed in ellipsoidal
        approximation by using small correction to the
        spherical approximation (see [3]_, eq. 26-28).

        References
        ----------
        .. [3] Jekeli, C. An analysis of vertical deflections derived
            from high-degree spherical harmonic models.
            Journal of Geodesy 73, 10–22 (1999).

        """

        dapot_lat = self._anomalous.derivative(position, 'lat', 'spherical')
        dapot_lon = self._anomalous.derivative(position, 'lon', 'spherical')
        dapot_rad = self._anomalous.derivative(position, 'r', 'spherical')

        ellharm = position.represent_as('ellipsoidalharmonic')
        geod = position.represent_as('geodetic')
        sph = position.represent_as('spherical')

        gamma = self._ell.normal_gravity(ellharm.rlat, ellharm.u_ax)

        denom_eta = gamma * sph.distance * _np.cos(sph.lat)
        eta = -(dapot_lon / denom_eta) * u.rad

        nu = geod.lat - sph.lat
        cnu = _np.cos(nu)
        snu = _np.sin(nu)

        xi = -(cnu * dapot_lat / sph.distance - snu * dapot_rad) / gamma * u.rad

        return _np.sqrt(eta**2 + xi**2)

    @u.quantity_input
    def vertical_deflection_ew(self, position) -> u.arcsec:
        """Return east-west component of the DoV.

        This is the east-west component of the gravimetric deflection of the vertical.
        It is the east-west component of the angle between the vector of gravity and
        the vector of normal gravity both at the same point (h,λ,φ).

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            East-west component of the DoV.

        Notes
        -----
        The gravimetric vertical deflection computed in ellipsoidal
        approximation by using small correction to the
        spherical approximation (see [3]_, eq. 26-28).

        References
        ----------
        .. [3] Jekeli, C. An analysis of vertical deflections derived
            from high-degree spherical harmonic models.
            Journal of Geodesy 73, 10–22 (1999).

        """

        dapot_lon = self._anomalous.derivative(position, 'lon', 'spherical')

        ellharm = position.represent_as('ellipsoidalharmonic')
        geod = position.represent_as('geodetic')
        sph = position.represent_as('spherical')

        gamma = self._ell.normal_gravity(ellharm.rlat, ellharm.u_ax)

        denom = gamma * sph.distance * _np.cos(sph.lat)
        eta = (-dapot_lon / denom) * u.rad

        return eta

    @u.quantity_input
    def vertical_deflection_ns(self, position) -> u.arcsec:
        """Return north-south component of the DoV.

        This is the north-south component of the gravimetric deflection of the vertical.
        It is the north-south component of the angle between the vector of gravity and
        the vector of normal gravity both at the same point (h,λ,φ).

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            North-south component of the DoV.

        Notes
        -----
        The gravimetric vertical deflection computed in ellipsoidal
        approximation by using small correction to the
        spherical approximation (see [3]_, eq. 26-28).

        References
        ----------
        .. [3] Jekeli, C. An analysis of vertical deflections derived
            from high-degree spherical harmonic models.
            Journal of Geodesy 73, 10–22 (1999).

        """

        dapot_lat = self._anomalous.derivative(position, 'lat', 'spherical')
        dapot_rad = self._anomalous.derivative(position, 'r', 'spherical')

        ellharm = position.represent_as('ellipsoidalharmonic')
        geod = position.represent_as('geodetic')
        sph = position.represent_as('spherical')

        gamma = self._ell.normal_gravity(ellharm.rlat, ellharm.u_ax)

        nu = geod.lat - sph.lat
        cnu = _np.cos(nu)
        snu = _np.sin(nu)

        xi = -(cnu * dapot_lat / sph.distance - snu * dapot_rad) / gamma * u.rad

        return xi


class SHGravPotential(_PotentialBase):

    @u.quantity_input
    def __init__(self,
                 coeffs: u.dimensionless_unscaled,
                 gm: u.m**3 / u.s**2, r0 : u.m, omega: 1 / u.s = None,
                 errors: bool = None, lmax: int = None, copy: bool = False):

        self._coeffs = _SHGravCoeffs.from_array(coeffs=coeffs, gm=gm, r0=r0,
                                                lmax=lmax, errors=errors,
                                                omega=omega, copy=True)

    @u.quantity_input
    def _potential(self, position) -> u.m**2 / u.s**2:
        """Return potential value.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Potential.
        """

        sph = position.represent_as('spherical')

        _, _, degrees, cosin, x, q = _expand.common_precompute(
            sph.lat, sph.lon, sph.distance, self._coeffs.r0, self._coeffs.lmax)
        args = (_expand.in_coeff_potential, _expand.sum_potential,
                self._coeffs.lmax, degrees, cosin, self._coeffs.coeffs)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / sph.distance

        out = _np.squeeze(self._coeffs.gm * ri * values)

        return out

    def _derivative(self, position, variable, coordinates):

        if coordinates is None:
            coordinates = position.representation_type.get_name()

        if coordinates == 'spherical':
            sph = position.represent_as('spherical')
            lat, _, degrees, cosin, x, q = _expand.common_precompute(
                sph.lat, sph.lon, sph.distance, self._coeffs.r0, self._coeffs.lmax)
            ri = 1 / sph.distance

            if variable in ('lat', 'latitude'):
                args = (_expand.in_coeff_lat_derivative, _expand.sum_lat_derivative,
                        self._coeffs.lmax, degrees, cosin, self._coeffs.coeffs)
                factor = self._coeffs.gm * ri * _np.cos(lat)
            elif variable in ('lon', 'longitude', 'long'):
                m_coeff = _np.tile(degrees, (self._coeffs.lmax + 1, 1))
                args = (_expand.in_coeff_lon_derivative, _expand.sum_lon_derivative,
                        self._coeffs.lmax, degrees, m_coeff, cosin,
                        self._coeffs.coeffs)
                factor = -self._coeffs.gm * ri
            elif variable in ('distance', 'radius', 'r', 'radial'):
                args = (_expand.in_coeff_r_derivative, _expand.sum_potential,
                        self._coeffs.lmax, degrees, cosin, self._coeffs.coeffs)
                factor = -self._coeffs.gm * ri**2

            values = _expand.expand_parallel(x, q, *args)
            out = _np.squeeze(factor * values)

        return out

    @u.quantity_input
    def r_derivative(self, position) -> u.m / u.s**2:
        """Return radial derivative of the potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Radial derivative.
        """
        return self._derivative(position, 'radius', 'spherical')

    @u.quantity_input
    def lat_derivative(self, position):
        """Return latitudinal derivative of the potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Latitudinal derivative.
        """
        return self._derivative(position, 'lat', 'spherical')

    @u.quantity_input
    def lon_derivative(self, position):
        """Return longitudinal derivative of the potential.

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        Returns
        -------
        ~astropy.units.Quantity
            Longitudinal derivative.
        """
        return self._derivative(position, 'lon', 'spherical')

    @u.quantity_input
    def _gradient_vector(self, position, coordinates):
        """Return gradient vector.

        The magnitude and the components of the gradient of the potential
        calculated on or above the ellipsoid without the centrifugal potential
        (eqs. 7 and 122 of STR09/02).

        Parameters
        ----------
        position : ~pygeoid.coordinates.frame.ECEF
            Position in the Earth-Centered-Earth-Fixed frame.

        """
        sph = position.represent_as('spherical')

        lat, _, degrees, cosin, x, q = _expand.common_precompute(
            sph.lat, sph.lon, sph.distance, self._coeffs.r0, self._coeffs.lmax)

        m_coeff = _np.tile(degrees, (self._coeffs.lmax + 1, 1))
        args = (_expand.in_coeff_gradient, _expand.sum_gradient,
                self._coeffs.lmax, degrees, m_coeff, cosin,
                self._coeffs.coeffs)

        values = _expand.expand_parallel(x, q, *args)

        ri = 1 / sph.distance
        gmri = self._coeffs.gm * ri
        clat = _np.cos(lat)

        lat_d = gmri * clat * values[:, :, 0]
        lon_d = -gmri * values[:, :, 1]
        rad_d = -self._coeffs.gm * ri**2 * values[:, :, 2]

        clati = _np.atleast_2d(1 / _np.ma.masked_values(clat, 0.0))
        clati = clati.filled(0.0)

        q1 = (ri * lat_d).squeeze()
        q2 = (clati * ri * lon_d).squeeze()
        q3 = rad_d.squeeze()

        return (q1, q2, q3)

    @u.quantity_input
    def _gradient(self, position, *args, **kwargs) -> u.m / u.s**2:
        return _np.linalg.norm(u.Quantity(
            self._gradient_vector(position, 'spherical')), axis=0)
