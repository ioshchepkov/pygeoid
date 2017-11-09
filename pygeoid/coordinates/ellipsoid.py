"""
Geometry of the reference ellipsoid
"""

import numpy as np
import pyproj as _proj
from pygeoid.constants import _2pi, _4pi

# update Proj ellipsoid parameters
_proj.pj_ellps.update({
    'PZ90': {'description': 'PZ-90', 'a': 6378136.0, 'rf': 298.25784},
    'GSK2011': {'description': 'GSK-2011', 'a': 6378136.5, 'rf': 298.2564151}
})

# default ellipsoid for geometrical (geodetic) applications
DEFAULT_ELLIPSOID = 'GRS80'


class Ellipsoid(_proj.Geod):
    """Class represents an ellipsoid of revolution and its geometry.

    This class intialize proj.Geod class from pyproj package, so any valid init
    string for Proj are accepted as arguments. See pyproj.Geod.__new__
    documentation (https://jswhit.github.io/pyproj/pyproj.Geod-class.html)
    for more information.

    Parameters
    ----------
    ellps : str, optional
        Ellipsoid name, most common ellipsoids are accepted. Default is
        'GRS80'.
    """
    # pylint: disable=R0904

    def __new__(cls, ellps=None, **kwargs):
        if not kwargs:
            if ellps in _proj.pj_ellps:
                kwargs['ellps'] = ellps
            elif ellps is None or ellps.lower() == 'default':
                kwargs['ellps'] = DEFAULT_ELLIPSOID
            else:
                raise ValueError(
                    'No ellipsoid with name {:%s}, possible values \
                        are:\n{:%s}'.format(ellps,
                                            _proj.pj_ellps.keys()))

        # define useful short-named attributes
        geod = _proj.Geod.__new__(cls, **kwargs)
        geod.e2 = geod.es  # eccentricity squared
        geod.e = np.sqrt(geod.e2)  # eccentricity
        geod.e12 = geod.e2 / (1 - geod.e2)  # 2nd eccentricity squared
        geod.e1 = np.sqrt(geod.e12)  # 2nd eccentricity

        return geod

    @property
    def equatorial_radius(self):
        """Return semi-major or equatorial axis radius :math:`a`, in meters."""
        return self.a

    @property
    def polar_radius(self):
        """Return semi-minor or polar axis radius :math:`b`, in meters."""
        return self.b

    @property
    def flattening(self):
        """Return flattening :math:`f`.

        The flattening of the ellipsoid is

        .. math::
            f = \\frac{a - b}{a},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.f

    @property
    def reciprocal_flattening(self):
        """Return reciprocal flattening :math:`1/f`."""
        return 1 / self.flattening

    @property
    def eccentricity(self):
        """Return first eccentricity :math:`e`.

        The first eccentricity of the ellipsoid is

        .. math::
            e = \\sqrt{\\frac{a^2 - b^2}{a^2}},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.e

    @property
    def eccentricity_squared(self):
        """Return first eccentricity squared :math:`e^2`."""
        return self.e2

    @property
    def second_eccentricity(self):
        """Return second eccentricity :math:`e'`.

        The second eccentricity of the ellipsoid is

        .. math::
            e' = \\sqrt{\\frac{a^2 - b^2}{b^2}}

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.e1

    @property
    def second_eccentricity_squared(self):
        """Return second eccentricity squared :math:`e'^2`."""
        return self.e12

    @property
    def linear_eccentricity(self):
        """Return linear eccentricity :math:`E`, in meters.

        The linear eccentricity of the ellipsoid is

        .. math::
            E = ae,

        where :math:`a` -- equatorial radius of the ellipsoid, :math:`e` --
        (first) eccentricity.
        """
        return self.equatorial_radius * self.eccentricity

    @property
    def polar_curvature_radius(self):
        """Return polar radius of curvature :math:`c`, in meters.

        The polar radius of curvature of the ellipsoid is

        .. math::
            c = \\frac{a^2}{b},

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return self.equatorial_radius**2 / self.polar_radius

    @property
    def quadrant_distance(self):
        """Return arc of meridian :math:`Q` from equator to pole, in meters.

        The arc length of meridian from equator to pole is

        .. math::
            Q = c\\frac{\pi}{2}\\left( 1 - \\frac{3}{4}e'^2 + \\frac{45}{64}e'^4
            +  \\frac{175}{256}e'^6 + \\frac{11025}{16384}e'^8\\right),

        where :math:`c` -- polar radius of curvature, :math:`e'` -- second
        eccentricity.
        """
        prc = self.polar_curvature_radius
        return prc * np.pi / 2 * (1 -
                                  3 / 4 * self.e12 + 45 / 64 * self.e12**2 -
                                  175 / 256 * self.e12 ** 3 +
                                  11025 / 16384 * self.e12**4)

    @property
    def surface_area(self):
        """Return surface area of the ellipsoid :math:`A`, in squared meters.

        The surface area of the ellipsoid is

        .. math::
            A = 2\pi a^2 \\left[1 + \\frac{1 - e^2}{2e} \ln{\\left(
            \\frac{1 + e}{1 - e}\\right)}\\right],

        where :math:`a` -- equatorial axis of the ellipsoid, :math:`e` --
        (first) eccentricity.
        """
        return _2pi * self.a**2 * (
            1 + 0.5 * (1 - self.e2) / self.e * np.log((1 +
                                                       self.e) / (1 - self.e)))

    @property
    def volume(self):
        """Return volume of the elliposid :math:`V`, in cubical meters.

        The volume of the ellipsoid is

        .. math::
            V = \\frac{4}{3}\pi a^2 b,

        where :math:`a` and :math:`b` -- equatorial and polar axis of the
        ellipsoid respectively.
        """
        return _4pi * self.a**2 * self.b / 3

    def mean_radius(self, kind='arithmetic'):
        """Return the radius of a sphere :math:`R`.

        Parameters
        ----------
        kind : {'arithmetic', 'same_area', 'same_volume'}, optional
            Controls what kind of radius is returned.

            * 'arithmetic' returns the arithmetic mean value :math:`R_m` of the 3 semi-axis
                of the ellipsoid.
            * 'same_area' returns the radius :math:`R_A` of the sphere with the same surface
                area as the ellipsoid.
            * 'same_volume' returns the radius :math:`R_V` of the sphere with the same
                volume as the ellipsoid.

            Default is 'arithmetic'.

        Returns
        -------
        float
            mean radius of the ellipsoid, in meters

        Note
        ----
        The arithmetic mean radius of the ellipsoid is

        .. math:: R_m = \\frac{2a + b}{2},

        where :math:`a` and :math:`b` are equatorial and polar axis of the
        ellipsoid respectively.

        A sphere with the same surface area as the elliposid has the radius

        .. math:: R_A = \sqrt{\\frac{A}{4\pi}},

        where :math:`A` is the surface area of the ellipsoid.

        A sphere with the same volume as the ellipsoid has the radius

        .. math:: R_V = a^2 b.
        """
        if kind == 'arithmetic':
            radius = (2 * self.a + self.b) / 3
        elif kind == 'same_area':
            radius = np.sqrt(self.surface_area / _4pi)
        elif kind == 'same_volume':
            radius = np.power(self.a**2 * self.b, 1 / 3)

        return radius

    #########################################################################
    # Auxiliary methods
    #########################################################################
    def _w(self, lat):
        """Return auxiliary function W.

        The auxiliary funtion :math:`W` defined as

        .. math::
        W = \sqrt{1 - e^2\sin^2{\phi}},

        where :math:`e` -- (first) eccentricity of the ellipsoid, :math:`\phi`
        -- geodetic latitude.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude, in radians.

        Returns
        -------
        float or array_like of floats
            Value of W.
        """
        return np.sqrt(1 - self.e2 * np.sin(lat) ** 2)

    def _v(self, lat):
        """Return auxiliary function V.

        The auxiliary funtion :math:`V` defined as

        .. math::
        V = \sqrt{1 + e'^2\cos^2{\phi}},

        where :math:`e'` -- second eccentricity of the ellipsoid, :math:`\phi`
        -- geodetic latitude.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude, in radians.

        Returns
        -------
        float or array_like of floats
            Value of V.
        """
        return np.sqrt(1 + self.e12 * np.cos(lat) ** 2)

    #########################################################################
    # Curvature
    #########################################################################
    def meridian_curvature_radius(self, lat):
        """Return radius of curvature of meridian normal section :math:`M`.

        The radius of curvature of meridian normal section is

        .. math::
            M = \\frac{c}{V^3},

        where :math:`c` -- polar radius of curvature, :math:`V` -- auxiliary
        function which depends on geodetic latitude.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude, in radians.

        Returns
        -------
        float or array_like of floats
            Value of the radius of curvature of meridian normal section M,
            in meters.
        """
        return self.polar_curvature_radius / self._v(lat) ** 3

    def prime_vertical_curvature_radius(self, lat):
        """Return radius of curvature of prime vertical normal section N.

        The radius of curvature of prime vertical is

        .. math::
            N = \\frac{c}{V},

        where :math:`c` -- polar radius of curvature, :math:`V` -- auxiliary
        function which depends on geodetic latitude.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude, in radians.

        Returns
        -------
        float or array_like of floats
            Value of the radius of curvature of prime vertical normal section N,
            in meters.
        """
        return self.polar_curvature_radius / self._v(lat)

    def gaussian_curvature_radius(self, lat, radians=False):
        """Return Gaussian radius of curvature, in meters

        The Gaussian radius of curvature is

        .. math::
            R = \\sqrt{MN},

        where :math:`M` -- radius of curvature of meridian normal section,
        :math:`N` -- radius of curvature of prime vertical.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.

        Returns
        -------
        float or array_like of floats
            Value of the Gaussian radius of curvature, in meters.
        """
        if not radians:
            lat = np.radians(lat)

        meridian_curv_radius = self.meridian_curvature_radius(lat)
        pvertical_curv_radius = self.prime_vertical_curvature_radius(lat)
        return np.sqrt(meridian_curv_radius * pvertical_curv_radius)

    def mean_curvature(self, lat, radians=False):
        """Return mean curvature

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.

        Returns
        -------
        float or array_like of floats
            Value of the mean curvature, in meters**-1.
        """
        if not radians:
            lat = np.radians(lat)

        return 0.5 * (1 / self.prime_vertical_curvature_radius(lat) +
                      1 / self.meridian_curvature_radius(lat))

    #########################################################################
    # Arc distanses, geodetic problems
    #########################################################################
    def meridian_arc_distance(self, lat1, lat2, radians=False):
        """Return the distance between two parallels lat1 and lat2

        Parameters
        ----------
        lat1 : float or array_like of floats
            Geodetic latitude of the first point.
        lat2 : float or array_like of floats
            Geodetic latitude of the second point.

        Returns
        -------
        float or array_like of floats
            The dustance between two parallels, in meters.
        """
        return self.inv(0., lat1, 0., lat2, radians=radians)

    def parallel_arc_distance(self, lat, lon1, lon2, radians=False):
        """Return the distance between two points on a parallel.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude of the parallel.
        lon1 : float or array_like of floats
            Geodetic longitude of the first point.
        lon2 : float or array_like of floats
            Geodetic longitude of the second point.

        Returns
        -------
        float or array_like of floats
            The dustance between two meridians, in meters.
        """
        return self.inv(lon1, lat, lon2, lat, radians=radians)

    #########################################################################
    # Radiuses
    #########################################################################
    def circle_radius(self, lat, radians=False):
        """Return the radius of the parallel lat, in meters."""
        if not radians:
            lat = np.radians(lat)
        return self.prime_vertical_curvature_radius(lat) * np.cos(lat)

    def polar_equation(self, lat, radians=False):
        """Return radius of the ellipsoid with respect to the origin.

        Parameters
        ----------
        lat : float or array_like of floats
            Geocentric latitude in radians.

        Returns
        -------
        float
            Geocentric radius of the parallel, in meters.
        """
        if not radians:
            lat = np.radians(lat)

        return (self.a * self.b) / (np.sqrt(self.a**2 * np.sin(lat)**2 +
                                            self.b**2 * np.cos(lat)**2))

    #########################################################################
    # Latitudes
    #########################################################################
    def geocentric_latitude(self, lat, radians=False):
        """Convert geodetic latitude to geocentric latitude.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.

        Returns
        -------
        float
            Geocentric (spherical) latitude, in radians.
        """
        if not radians:
            lat = np.radians(lat)
        return np.arctan((1 - self.f)**2 * np.tan(lat))

    def reduced_latitude(self, lat, radians=False):
        """Convert geodetic latitude to reduced latitude.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.

        Returns
        -------
        float
            Reduced latitude, in radians.
        """
        if not radians:
            lat = np.radians(lat)
        return np.arctan((1 - self.f) * np.tan(lat))
