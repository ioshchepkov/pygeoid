"""Geometry of the reference ellipsoid.

"""

import numpy as _np
import pyproj as _proj
import astropy.units as u
from pygeoid.constants import _2pi, _4pi

# default ellipsoid for geometrical (geodetic) applications
DEFAULT_ELLIPSOID = 'GRS80'


class Ellipsoid:
    """Class represents an ellipsoid of revolution and its geometry.

    This class uses proj.Geod class from pyproj package, so any valid init
    string for Proj are accepted as arguments. See `pyproj.Geod.__new__`
    documentation (https://pyproj4.github.io/pyproj/stable/api/geod.html)
    for more information.

    Parameters
    ----------
    ellps : str, optional
        Ellipsoid name, most common ellipsoids are accepted.
        Default is 'GRS80'.
    """

    def __init__(self, ellps: 'str' = None, **kwargs):
        if not kwargs:
            if ellps in _proj.pj_ellps:
                kwargs['ellps'] = ellps
            elif ellps is None or ellps.lower() == 'default':
                kwargs['ellps'] = DEFAULT_ELLIPSOID
            else:
                raise ValueError(
                    'No ellipsoid with name {0}, possible values \
                        are:\n{1}'.format(ellps, _proj.pj_ellps.keys()))
        # else:
            # TODO: Check if all parameters are in SI units
        #    pass

        # define useful short-named attributes
        self.geod = _proj.Geod(**kwargs)
        self.a = self.geod.a * u.m
        self.b = self.geod.b * u.m
        self.f = self.geod.f * u.dimensionless_unscaled  # flattening
        self.e2 = _np.float64(self.geod.es) * u.dimensionless_unscaled  # eccentricity squared
        self.e = _np.sqrt(self.e2) * u.dimensionless_unscaled  # eccentricity
        self.e12 = self.e2 / (1 - self.e2) * u.dimensionless_unscaled  # 2nd eccentricity squared
        self.e1 = _np.sqrt(self.e12) * u.dimensionless_unscaled  # 2nd eccentricity

    @property
    def equatorial_radius(self):
        """Return semi-major or equatorial axis radius, in metres.

        """
        return self.a

    @property
    def polar_radius(self):
        """Return semi-minor or polar axis radius, in metres.

        """
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
        """Return reciprocal (inverse) flattening.

        """
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
        """Return first eccentricity squared.

        """
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
        """Return second eccentricity squared.

        """
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
        return prc * _np.pi / 2 * (1 -
                                   3 / 4 * self.e12 + 45 / 64 * self.e12**2 -
                                   175 / 256 * self.e12 ** 3 +
                                   11025 / 16384 * self.e12**4)

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
        return _2pi * self.a**2 * (
            1 + 0.5 * (1 - self.e2) / self.e * _np.log((1 +
                                                        self.e) / (1 -
                                                                   self.e)))

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
        return _4pi * self.a**2 * self.b / 3

    def mean_radius(self, kind: str = 'arithmetic'):
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

        .. math:: R_m = \frac{2a + b}{2},

        where :math:`a` and :math:`b` are equatorial and polar axis of the
        ellipsoid respectively.

        A sphere with the same surface area as the elliposid has the radius

        .. math:: R_A = \sqrt{\frac{A}{4\pi}},

        where :math:`A` is the surface area of the ellipsoid.

        A sphere with the same volume as the ellipsoid has the radius

        .. math:: R_V = a^2 b.

        """
        if kind == 'arithmetic':
            radius = (2 * self.a + self.b) / 3
        elif kind == 'same_area':
            radius = _np.sqrt(self.surface_area / _4pi)
        elif kind == 'same_volume':
            radius = _np.power(self.a**2 * self.b, 1 / 3)
        else:
            raise ValueError('Not a valid `kind` of the radius.')

        return radius

    #########################################################################
    # Auxiliary methods
    #########################################################################
    @u.quantity_input
    def _w(self, lat: u.deg) -> u.dimensionless_unscaled:
        r"""Return auxiliary function W.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
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
        lat : ~astropy.units.Quantity
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
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
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
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
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
    def mean_curvature(self, lat: u.deg) -> 1 / u.m:
        r"""Return mean curvature, in inverse metres.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
            Value of the mean curvature.

        Notes
        -----
        The mean curvature is :math:`1/\sqrt{MN}`, where
        :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.
        """
        meridian_curv_radius = self.meridian_curvature_radius(lat)
        pvertical_curv_radius = self.prime_vertical_curvature_radius(lat)
        return 1 / _np.sqrt(meridian_curv_radius * pvertical_curv_radius)

    @u.quantity_input
    def gaussian_curvature(self, lat: u.deg) -> 1 / u.m:
        """Return Gaussian curvature, in inverse metres.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
            Value of the Gaussian radius of curvature.

        Notes
        -----
        The Gaussian curvature is :math:`1/MN`, where
        :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.
        """
        meridian_curv_radius = self.meridian_curvature_radius(lat)
        pvertical_curv_radius = self.prime_vertical_curvature_radius(lat)
        return _np.sqrt(meridian_curv_radius * pvertical_curv_radius)

    @u.quantity_input
    def average_curvature(self, lat: u.deg) -> 1 / u.m:
        r"""Return average curvature, in inverse metres.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
            Value of the average curvature.

        Notes
        -----
        The average curvature is

        .. math:: \frac{1}{2} \left( \frac{1}{M} + \frac{1}{N} \right),

        where :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.

        """
        return 0.5 * (1 / self.prime_vertical_curvature_radius(lat) +
                      1 / self.meridian_curvature_radius(lat))

    #########################################################################
    # Arc distances, geodetic problems
    #########################################################################

    @u.quantity_input
    def meridian_arc_distance(self, lat1: u.deg, lat2: u.deg) -> u.m:
        """Return the distance between two parallels `lat1` and `lat2`.

        Parameters
        ----------
        lat1 : ~astropy.units.Quantity
            Geodetic latitude of the first point.
        lat2 : ~astropy.units.Quantity
            Geodetic latitude of the second point.

        Returns
        -------
        ~astropy.units.Quantity
            The distance between two parallels.

        """
        return self.inv(lat1, 0. * u.deg, lat2, 0. * u.deg)[-1]

    @u.quantity_input
    def parallel_arc_distance(self, lat: u.deg, lon1: u.deg, lon2: u.deg):
        """Return the distance between two points on a parallel.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude of the parallel.
        lon1 : ~astropy.units.Quantity
            Geodetic longitude of the first point.
        lon2 : ~astropy.units.Quantity
            Geodetic longitude of the second point.

        Returns
        -------
        ~astropy.units.Quantity
            The distance between two meridians along the parallel.
        """
        return self.circle_radius(lat) * (lon2 - lon1).to('radian')

    @u.quantity_input
    def fwd(self, lat: u.deg, lon: u.deg, azimuth: u.deg, distance: u.m):
        """Solve forward geodetic problem.

        Returns latitudes, longitudes and back azimuths of terminus points
        given latitudes `lat` and longitudes `lon` of initial points, plus
        forward `azimuth`s and `distance`s.

        This method use `pyproj.Geod.fwd` as a backend.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude of the initial point.
        lon : ~astropy.units.Quantity
            Longitude of the initial point.
        azimuth : ~astropy.units.Quantity
            Geodetic azimuth.
        distance : ~astropy.units.Quantity
            Distance.

        Returns
        -------
        lat : ~astropy.units.Quantity
            Geodetic latitude of the terminus point.
        lon : ~astropy.units.Quantity
            Longitude of the terminus point.
        back_azimuth : ~astropy.units.Quantity
            Back geodetic azimuth.
        """
        out_lon, out_lat, out_baz = self.geod.fwd(lon.to('radian').value,
                                                  lat.to('radian').value,
                                                  azimuth.to('radian').value,
                                                  distance.to('m').value,
                                                  radians=True)
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
        lat1 : ~astropy.units.Quantity
            Geodetic latitude of the initial point.
        lon1 : ~astropy.units.Quantity
            Longitude of the initial point.
        lat2 : ~astropy.units.Quantity
            Geodetic latitude of the terminus point.
        lon2 : ~astropy.units.Quantity
            Longitude of the terminus point.

        Returns
        -------
        azimuth : ~astropy.units.Quantity
            Geodetic azimuth.
        back_azimuth : ~astropy.units.Quantity
            Back geodetic azimuth.
        distance : ~astropy.units.Quantity
            Distance, in metres.
        """
        azimuth, back_azimuth, distance = self.geod.inv(
            lon1.to('radian').value,
            lat1.to('radian').value,
            lon2.to('radian').value,
            lat2.to('radian').value, radians=True)

        return azimuth * u.rad, back_azimuth * u.rad, distance * u.m

    @u.quantity_input
    def npts(self, lat1: u.deg, lon1: u.deg,
             lat2: u.deg, lon2: u.deg, npts: int) -> u.deg:
        """Return equaly spaced points along geodesic line.

        Given a single initial point and terminus point (specified by
        `lat1`, `lon1` and `lat2`, `lon2`), returns a list of
        longitude/latitude pairs describing npts equally spaced
        intermediate points along the geodesic between the initial
        and terminus points.

        This method use `pyproj.Geod.npts` as a backend.

        Parameters
        ----------
        lat1 : ~astropy.units.Quantity
            Geodetic latitude of the initial point.
        lon1 : ~astropy.units.Quantity
            Longitude of the initial point.
        lat2 : ~astropy.units.Quantity
            Geodetic latitude of the terminus point.
        lon2 : ~astropy.units.Quantity
            Longitude of the terminus point.
        npts : int
            Number of intermediate points.

        Returns
        -------
        points : ~astropy.units.Quantity list of tuples
            List of latitudes and longitudes of the intermediate points.
        """
        points = self.geod.npts(
            lon1.to('radian').value,
            lat1.to('radian').value,
            lon2.to('radian').value,
            lat2.to('radian').value, npts, radians=True)

        return points * u.rad

    #########################################################################
    # Radii
    #########################################################################
    @u.quantity_input
    def circle_radius(self, lat: u.deg) -> u.m:
        r"""Return the radius of the parallel, in metres.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
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
        lat : ~astropy.units.Quantity
            **Geocentric** latitude.

        Returns
        -------
        ~astropy.units.Quantity
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
        return (self.a * self.b) / (_np.sqrt(self.a**2 * _np.sin(lat)**2 +
                                             self.b**2 * _np.cos(lat)**2))

    #########################################################################
    # Latitudes
    #########################################################################
    @u.quantity_input
    def geocentric_latitude(self, lat: u.deg) -> u.deg:
        r"""Convert geodetic latitude to geocentric latitude.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
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
        geoc_lat = _np.arctan((1 - self.f)**2 * _np.tan(lat))

        return geoc_lat

    @u.quantity_input
    def reduced_latitude(self, lat: u.deg) -> u.deg:
        r"""Convert geodetic latitude to reduced (parametric) latitude.

        Parameters
        ----------
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
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
        lat : ~astropy.units.Quantity
            Geodetic latitude.

        Returns
        -------
        ~astropy.units.Quantity
            Authalic latitude.

        """

        def q(lat):
            slat = _np.sin(lat)
            log = 0.5 / self.e * _np.log((1 - self.e * slat) / (1 + self.e * slat))
            return (1 - self.e2) * (slat / (1 - self.e2 * slat**2) - log)

        auth_lat = _np.arcsin(q(lat) / q(_np.pi / 2))

        return auth_lat
