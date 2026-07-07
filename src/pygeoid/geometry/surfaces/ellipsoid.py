"""Geometry of reference ellipsoids."""

import functools as _functools

import numpy as _np
import pyproj as _proj

from pygeoid.conventions import units as u
from pygeoid.geometry.surfaces.base import Surface

# default ellipsoid for geometrical (geodetic) applications
DEFAULT_ELLIPSOID = "GRS80"

_GEOD_LENGTH_PARAMETERS = {"a", "b"}
_GEOD_DIMENSIONLESS_PARAMETERS = {"e", "f", "rf", "es"}


def get_ellps_map():
    """Return pyproj ellipsoid definitions with unit-aware parameters.

    This wraps `pyproj.get_ellps_map` and returns a new dictionary where
    semi-axis parameters are quantities in metres and flattening parameters are
    dimensionless quantities. Descriptions are preserved as strings.
    """
    return {
        ellps: {
            key: _ellipsoid_parameter_with_units(key, value)
            for key, value in params.items()
        }
        for ellps, params in _proj.get_ellps_map().items()
    }


def _ellipsoid_parameter_with_units(key, value):
    if key in _GEOD_LENGTH_PARAMETERS:
        return value * u.m
    if key in _GEOD_DIMENSIONLESS_PARAMETERS:
        return value * u.dimensionless_unscaled
    return value


class Ellipsoid(Surface):
    """Ellipsoid of revolution and its geometric properties.

    This class wraps `pyproj.Geod` for geodesic calculations and exposes
    common ellipsoid parameters and derived geometric quantities as unit-aware
    properties.

    Named ellipsoids can be selected with ``ellps``. If no name or custom
    parameters are given, the default named ellipsoid is ``GRS80``. ``None``
    and ``"default"`` also select ``GRS80``.

    Custom ellipsoids can be created with keyword parameters accepted by
    `pyproj.Geod`, but unlike `pyproj.Geod`, numeric custom values must be
    quantities with proper units. Length parameters such as ``a`` and ``b`` are
    converted to metres; dimensionless parameters such as ``f``, ``rf``, ``e`` and
    ``es`` are converted to plain dimensionless values.

    Use `get_ellps_map` to get PyProj's named ellipsoid definitions with units
    attached to numeric parameters. Those definitions can be passed directly to
    this constructor with ``Ellipsoid(**params)``.

    Parameters
    ----------
    ellps : str or None, optional
        Named ellipsoid understood by `pyproj.Geod`.
    **kwargs
        Custom ellipsoid parameters with units accepted by `pyproj.Geod`,
        including definitions returned by `get_ellps_map`.
    """

    def __init__(self, ellps: str | None = None, **kwargs):
        if kwargs:
            kwargs = self._geod_kwargs_from_quantities(kwargs)
            if ellps is not None:
                kwargs["ellps"] = self._ellipsoid_name(ellps)
        else:
            kwargs["ellps"] = self._ellipsoid_name(ellps)

        self._geod = _proj.Geod(**kwargs)

    @classmethod
    def from_pyproj_crs(cls, crs):
        """Create an ellipsoid from a pyproj-compatible CRS definition.

        Parameters
        ----------
        crs
            Any input accepted by `pyproj.CRS.from_user_input`, such as an EPSG
            code, authority string, WKT, PROJ string, or `pyproj.CRS` instance.

        Returns
        -------
        Ellipsoid
            Ellipsoid initialized from the CRS geodesic definition.
        """
        self = cls.__new__(cls)
        self._geod = _proj.CRS.from_user_input(crs).get_geod()
        return self

    @property
    def geod(self):
        """Read-only `pyproj.Geod` backend."""
        return self._geod

    def to_proj_geod(self):
        """Return a `pyproj.Geod` initialized from the original parameters."""
        return _proj.Geod(self.geod.initstring)

    @staticmethod
    def _ellipsoid_name(ellps):
        if ellps in _proj.pj_ellps:
            return ellps
        if ellps is None or ellps.lower() == "default":
            return DEFAULT_ELLIPSOID
        raise ValueError(
            f"No ellipsoid with name {ellps}, possible values \
                are:\n{_proj.pj_ellps.keys()}"
        )

    @staticmethod
    def _geod_kwargs_from_quantities(kwargs):
        geod_kwargs = {}
        for key, value in kwargs.items():
            if key == "description":
                geod_kwargs[key] = value
                continue

            if not isinstance(value, u.Quantity):
                raise TypeError(f"`{key}` must be a quantity with units.")

            if key in _GEOD_LENGTH_PARAMETERS:
                if not value.unit.is_equivalent(u.m):
                    raise u.UnitTypeError(f"`{key}` must have length units.")
                geod_kwargs[key] = value.to(u.m).value
            elif key in _GEOD_DIMENSIONLESS_PARAMETERS:
                if not value.unit.is_equivalent(u.dimensionless_unscaled):
                    raise u.UnitTypeError(f"`{key}` must be dimensionless.")
                geod_kwargs[key] = value.to(u.dimensionless_unscaled).value
            else:
                geod_kwargs[key] = value.si.value

        return geod_kwargs

    @_functools.cached_property
    def a(self):
        """Return semi-major or equatorial axis radius, in metres."""
        return self.geod.a * u.m

    @_functools.cached_property
    def b(self):
        """Return semi-minor or polar axis radius, in metres."""
        return self.geod.b * u.m

    @_functools.cached_property
    def f(self):
        """Return flattening of the ellipsoid."""
        return self.geod.f * u.dimensionless_unscaled

    @_functools.cached_property
    def e2(self):
        """Return first eccentricity squared."""
        return _np.float64(self.geod.es) * u.dimensionless_unscaled

    @_functools.cached_property
    def e(self):
        """Return first eccentricity."""
        return _np.sqrt(self.e2) * u.dimensionless_unscaled

    @_functools.cached_property
    def e12(self):
        """Return second eccentricity squared."""
        return self.e2 / (1 - self.e2) * u.dimensionless_unscaled

    @_functools.cached_property
    def e1(self):
        """Return second eccentricity."""
        return _np.sqrt(self.e12) * u.dimensionless_unscaled

    @property
    def equatorial_radius(self):
        """Return semi-major or equatorial axis radius, in metres."""
        return self.a

    @property
    def polar_radius(self):
        """Return semi-minor or polar axis radius, in metres."""
        return self.b

    @property
    def flattening(self):
        r"""Return flattening of the ellipsoid.

        Notes
        -----
        The flattening of the ellipsoid :math:`f` is

        .. math::
            f = \frac{a - b}{a},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.f

    @property
    def reciprocal_flattening(self):
        """Return reciprocal (inverse) flattening."""
        return 1 / self.flattening

    @property
    def eccentricity(self):
        r"""Return first eccentricity.

        Notes
        -----
        The first eccentricity of the ellipsoid :math:`e` is

        .. math::
            e = \sqrt{\frac{a^2 - b^2}{a^2}},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.e

    @property
    def eccentricity_squared(self):
        """Return first eccentricity squared."""
        return self.e2

    @property
    def second_eccentricity(self):
        r"""Return second eccentricity.

        Notes
        -----
        The second eccentricity of the ellipsoid :math:`e'` is

        .. math::
            e' = \sqrt{\frac{a^2 - b^2}{b^2}}

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.e1

    @property
    def second_eccentricity_squared(self):
        """Return second eccentricity squared."""
        return self.e12

    @property
    def linear_eccentricity(self):
        """Return linear eccentricity, in metres.

        Notes
        -----
        The linear eccentricity of the ellipsoid :math:`E` is

        .. math::
            E = ae,

        where :math:`a` -- equatorial radius of the ellipsoid, :math:`e` --
        (first) eccentricity.
        """
        return self.equatorial_radius * self.eccentricity

    @property
    def polar_curvature_radius(self):
        r"""Return polar radius of curvature, in metres.

        Notes
        -----
        The polar radius of curvature of the ellipsoid :math:`c` is

        .. math::
            c = \frac{a^2}{b},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.equatorial_radius**2 / self.polar_radius

    @property
    def quadrant_distance(self):
        r"""Return arc of meridian from equator to pole, in metres.

        Notes
        -----
        The arc length of meridian from equator to pole is

        .. math::
            Q = c\frac{\pi}{2}\left( 1 - \frac{3}{4}e'^2 +
            \frac{45}{64}e'^4 +  \frac{175}{256}e'^6 +
            \frac{11025}{16384}e'^8\right),

        where :math:`c` -- polar radius of curvature, :math:`e'` -- second
        eccentricity.
        """
        prc = self.polar_curvature_radius
        return (
            prc
            * _np.pi
            / 2
            * (
                1
                - 3 / 4 * self.e12
                + 45 / 64 * self.e12**2
                - 175 / 256 * self.e12**3
                + 11025 / 16384 * self.e12**4
            )
        )

    @property
    def surface_area(self):
        r"""Return surface area of the ellipsoid, in squared metres.

        Notes
        -----
        The surface area of the ellipsoid is

        .. math::
            A = 2\pi a^2 \left[1 + \frac{1 - e^2}{2e} \ln{\left(
            \frac{1 + e}{1 - e}\right)}\right],

        where :math:`a` -- equatorial axis of the ellipsoid, :math:`e` --
        (first) eccentricity.
        """
        return (
            2
            * _np.pi
            * self.a**2
            * (1 + 0.5 * (1 - self.e2) / self.e * _np.log((1 + self.e) / (1 - self.e)))
        )

    @property
    def volume(self):
        r"""Return volume of the ellipsoid, in cubical metres.

        Notes
        -----
        The volume of the ellipsoid is

        .. math::
            V = \frac{4}{3}\pi a^2 b,

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.

        """
        return 4 * _np.pi * self.a**2 * self.b / 3

    def mean_radius(self, kind: str = "arithmetic"):
        r"""Return the radius of a sphere.

        Parameters
        ----------
        kind : {'arithmetic', 'same_area', 'same_volume'}, optional
            Controls what kind of radius is returned.

            * 'arithmetic' returns the arithmetic mean value
                :math:`R_m` of the 3 semi-axis of the ellipsoid.
            * 'same_area' returns the authalic radius :math:`R_A` of
                the sphere with the same surface
                area as the ellipsoid.
            * 'same_volume' returns the radius :math:`R_V` of
                the sphere with the same volume as the ellipsoid.

            Default is 'arithmetic'.

        Returns
        -------
        float
            Mean radius of the ellipsoid, in metres.

        Notes
        -----
        The arithmetic mean radius of the ellipsoid is

        .. math:: R_m = \frac{2a + b}{3},

        where :math:`a` and :math:`b` are equatorial and polar axis of the
        ellipsoid respectively.

        A sphere with the same surface area as the elliposid has the radius

        .. math:: R_A = \sqrt{\frac{A}{4\pi}},

        where :math:`A` is the surface area of the ellipsoid.

        A sphere with the same volume as the ellipsoid has the radius

        .. math:: R_V = \sqrt[3]{a^2 b}.

        """
        if kind == "arithmetic":
            radius = (2 * self.a + self.b) / 3
        elif kind == "same_area":
            radius = _np.sqrt(self.surface_area / (4 * _np.pi))
        elif kind == "same_volume":
            radius = _np.power(self.a**2 * self.b, 1 / 3)
        else:
            raise ValueError("Not a valid `kind` of the radius.")

        return radius

    #########################################################################
    # Auxiliary methods
    #########################################################################
    @u.quantity_input
    def _w(self, lat: u.deg) -> u.dimensionless_unscaled:
        r"""Return auxiliary function W.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        float or array_like of floats
            Value of W.

        Notes
        -----
        The auxiliary function :math:`W` defined as

        .. math::
        W = \sqrt{1 - e^2\sin^2{\phi}},

        where :math:`e` -- (first) eccentricity of the ellipsoid, :math:`\phi`
        -- geodetic latitude.
        """
        return _np.sqrt(1 - self.e2 * _np.sin(lat) ** 2)

    @u.quantity_input
    def _v(self, lat: u.deg) -> u.dimensionless_unscaled:
        r"""Return auxiliary function V.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        float or array_like of floats
            Value of V.

        Notes
        -----
        The auxiliary function :math:`V` defined as

        .. math::
        V = \sqrt{1 + e'^2\cos^2{\phi}},

        where :math:`e'` -- second eccentricity of the ellipsoid, :math:`\phi`
        -- geodetic latitude.
        """
        return _np.sqrt(1 + self.e12 * _np.cos(lat) ** 2)

    #########################################################################
    # Curvature
    #########################################################################
    @u.quantity_input
    def meridian_curvature_radius(self, lat: u.deg) -> u.m:
        r"""Return radius of curvature of meridian normal section.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Value of the radius of curvature of meridian normal section.

        Notes
        -----
        The radius of curvature of meridian normal section :math:`M` is

        .. math::
            M = \frac{c}{V^3},

        where :math:`c` -- polar radius of curvature, :math:`V` -- auxiliary
        function which depends on geodetic latitude.

        """
        return self.polar_curvature_radius / self._v(lat) ** 3

    @u.quantity_input
    def prime_vertical_curvature_radius(self, lat: u.deg) -> u.m:
        r"""Return radius of curvature of prime vertical normal section.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Value of the radius of curvature of prime vertical
            normal section.

        Notes
        -----
        The radius of curvature of prime vertical :math:`N` is

        .. math::
            N = \frac{c}{V},

        where :math:`c` -- polar radius of curvature, :math:`V` -- auxiliary
        function which depends on geodetic latitude.
        """
        return self.polar_curvature_radius / self._v(lat)

    @u.quantity_input
    def _principal_curvature_radii(self, lat: u.deg):
        """Return both meridian and prime vertical curvature radii."""
        return (
            self.meridian_curvature_radius(lat),
            self.prime_vertical_curvature_radius(lat),
        )

    @u.quantity_input
    def mean_curvature(self, lat: u.deg) -> 1 / u.m:
        r"""Return mean curvature, in inverse metres.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Value of the mean curvature.

        Notes
        -----
        The mean curvature is :math:`1/\sqrt{MN}`, where
        :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.
        """
        m_rad, n_rad = self._principal_curvature_radii(lat)
        return 1 / _np.sqrt(m_rad * n_rad)

    @u.quantity_input
    def gaussian_curvature(self, lat: u.deg) -> 1 / u.m**2:
        """Return Gaussian curvature, in inverse squared metres.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Value of the Gaussian curvature.

        Notes
        -----
        The Gaussian curvature is :math:`1/MN`, where
        :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.
        """
        m_rad, n_rad = self._principal_curvature_radii(lat)
        return 1 / (m_rad * n_rad)

    @u.quantity_input
    def average_curvature(self, lat: u.deg) -> 1 / u.m:
        r"""Return average curvature, in inverse metres.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Value of the average curvature.

        Notes
        -----
        The average curvature is

        .. math:: \frac{1}{2} \left( \frac{1}{M} + \frac{1}{N} \right),

        where :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.

        """
        m_rad, n_rad = self._principal_curvature_radii(lat)
        return 0.5 * (1 / m_rad + 1 / n_rad)

    #########################################################################
    # Arc distances, geodetic problems
    #########################################################################

    @u.quantity_input
    def meridian_arc_distance(self, lat1: u.deg, lat2: u.deg) -> u.m:
        """Return the distance between two parallels `lat1` and `lat2`.

        Parameters
        ----------
        lat1 : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the first point.
        lat2 : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the second point.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            The distance between two parallels.

        """
        return self.inv(lat1, 0.0 * u.deg, lat2, 0.0 * u.deg)[-1]

    @u.quantity_input
    def parallel_arc_distance(self, lat: u.deg, lon1: u.deg, lon2: u.deg):
        """Return the distance between two points on a parallel.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the parallel.
        lon1 : ~pygeoid.conventions.units.Quantity
            Geodetic longitude of the first point.
        lon2 : ~pygeoid.conventions.units.Quantity
            Geodetic longitude of the second point.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            The distance between two meridians along the parallel.
        """
        return self.circle_radius(lat) * (lon2 - lon1).to("radian")

    @u.quantity_input
    def fwd(self, lat: u.deg, lon: u.deg, azimuth: u.deg, distance: u.m):
        """Solve forward geodetic problem.

        Returns latitudes, longitudes and back azimuths of terminus points
        given latitudes ``lat`` and longitudes ``lon`` of initial points, plus
        forward ``azimuth`` values and ``distance`` values.

        This method use `pyproj.Geod.fwd` as a backend.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the initial point.
        lon : ~pygeoid.conventions.units.Quantity
            Longitude of the initial point.
        azimuth : ~pygeoid.conventions.units.Quantity
            Geodetic azimuth.
        distance : ~pygeoid.conventions.units.Quantity
            Distance.

        Returns
        -------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the terminus point.
        lon : ~pygeoid.conventions.units.Quantity
            Longitude of the terminus point.
        back_azimuth : ~pygeoid.conventions.units.Quantity
            Back geodetic azimuth.
        """
        out_lon, out_lat, out_baz = self.geod.fwd(
            lon.to("radian").value,
            lat.to("radian").value,
            azimuth.to("radian").value,
            distance.to("m").value,
            radians=True,
        )
        return out_lat * u.rad, out_lon * u.rad, out_baz * u.rad

    @u.quantity_input
    def inv(self, lat1: u.deg, lon1: u.deg, lat2: u.deg, lon2: u.deg):
        """Solve inverse geodetic problem.

        Returns forward and back azimuths, plus distances between initial
        points (specified by `lat1`, `lon1`) and terminus points (specified by
        `lat1`, `lon2`).

        This method use `pyproj.Geod.inv` as a backend.

        Parameters
        ----------
        lat1 : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the initial point.
        lon1 : ~pygeoid.conventions.units.Quantity
            Longitude of the initial point.
        lat2 : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the terminus point.
        lon2 : ~pygeoid.conventions.units.Quantity
            Longitude of the terminus point.

        Returns
        -------
        azimuth : ~pygeoid.conventions.units.Quantity
            Geodetic azimuth.
        back_azimuth : ~pygeoid.conventions.units.Quantity
            Back geodetic azimuth.
        distance : ~pygeoid.conventions.units.Quantity
            Distance, in metres.
        """
        azimuth, back_azimuth, distance = self.geod.inv(
            lon1.to("radian").value,
            lat1.to("radian").value,
            lon2.to("radian").value,
            lat2.to("radian").value,
            radians=True,
        )

        return azimuth * u.rad, back_azimuth * u.rad, distance * u.m

    @u.quantity_input
    def npts(
        self, lat1: u.deg, lon1: u.deg, lat2: u.deg, lon2: u.deg, npts: int
    ) -> u.deg:
        """Return equaly spaced points along geodesic line.

        Given a single initial point and terminus point (specified by
        `lat1`, `lon1` and `lat2`, `lon2`), returns a list of
        longitude/latitude pairs describing npts equally spaced
        intermediate points along the geodesic between the initial
        and terminus points.

        This method use `pyproj.Geod.npts` as a backend.

        Parameters
        ----------
        lat1 : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the initial point.
        lon1 : ~pygeoid.conventions.units.Quantity
            Longitude of the initial point.
        lat2 : ~pygeoid.conventions.units.Quantity
            Geodetic latitude of the terminus point.
        lon2 : ~pygeoid.conventions.units.Quantity
            Longitude of the terminus point.
        npts : int
            Number of intermediate points.

        Returns
        -------
        points : ~pygeoid.conventions.units.Quantity list of tuples
            List of latitudes and longitudes of the intermediate points.
        """
        points = self.geod.npts(
            lon1.to("radian").value,
            lat1.to("radian").value,
            lon2.to("radian").value,
            lat2.to("radian").value,
            npts,
            radians=True,
        )

        return points * u.rad

    #########################################################################
    # Radii
    #########################################################################
    @u.quantity_input
    def circle_radius(self, lat: u.deg) -> u.m:
        r"""Return the radius of the parallel, in metres.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Notes
        -----
        The radius of the parallel :math:`\phi` is

        .. math::
            r_\phi = N \cos{\phi},

        where :math:`N` -- radius of curvature of prime vertical, :math:`\phi`
        -- geodetic latitude.

        """
        return self.prime_vertical_curvature_radius(lat) * _np.cos(lat)

    @u.quantity_input
    def polar_equation(self, lat: u.deg) -> u.m:
        r"""Return radius of the ellipsoid with respect to the origin.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            **Geocentric** latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Geocentric radius of the parallel.

        Notes
        -----
        The polar equation of the ellipsoid is

        .. math::
            r = \frac{ab}{\sqrt{a^2\sin^2{\vartheta} +
            b^2\cos^2{\vartheta}}},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively, :math:`\vartheta` -- geocentric latitude.
        """
        return (self.a * self.b) / (
            _np.sqrt(self.a**2 * _np.sin(lat) ** 2 + self.b**2 * _np.cos(lat) ** 2)
        )

    #########################################################################
    # Latitudes
    #########################################################################
    @u.quantity_input
    def geocentric_latitude(self, lat: u.deg) -> u.deg:
        r"""Convert geodetic latitude to geocentric latitude.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Geocentric (spherical) latitude.

        Notes
        -----
        The relationship between geodetic :math:`\phi` and geocentric
        :math:`\vartheta` latitudes is

        .. math::
            \vartheta = \tan^{-1}{\left(\left(1 -
            f\right)^2\tan\phi\right)},

        where :math:`f` -- flattening of the ellipsoid.
        """
        geoc_lat = _np.arctan((1 - self.f) ** 2 * _np.tan(lat))

        return geoc_lat

    @u.quantity_input
    def reduced_latitude(self, lat: u.deg) -> u.deg:
        r"""Convert geodetic latitude to reduced (parametric) latitude.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Reduced latitude.

        Notes
        -----
        The relationship between geodetic :math:`\phi` and reduced
        :math:`\beta` latitudes is

        .. math::
            \beta = \tan^{-1}{\left(\left(1 - f\right)\tan\phi\right)},

        where :math:`f` -- flattening of the ellipsoid.
        """
        red_lat = _np.arctan((1 - self.f) * _np.tan(lat))

        return red_lat

    @u.quantity_input
    def authalic_latitude(self, lat: u.deg) -> u.deg:
        r"""Convert geodetic latitude to authalic latitude.

        Authalic latitude will return a geocentric latitude on a sphere having
        the same surface area as the ellipsoid. It will preserve areas with
        relative to the ellipsoid. The authalic radius can be
        calculated from `mean_radius(kind='same_area')` method.

        Parameters
        ----------
        lat : ~pygeoid.conventions.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~pygeoid.conventions.units.Quantity
            Authalic latitude.

        """

        def q(lat):
            slat = _np.sin(lat)
            log = 0.5 / self.e * _np.log((1 - self.e * slat) / (1 + self.e * slat))
            return (1 - self.e2) * (slat / (1 - self.e2 * slat**2) - log)

        auth_lat = _np.arcsin(q(lat) / q(_np.pi / 2))

        return auth_lat
