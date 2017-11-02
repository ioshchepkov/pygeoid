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


class _Ellipse:
    """Class representing an ellipse geometry.

    The class is used as a parent class for Ellipsoid and basically not for a direct use.
    """

    def __init__(self, a, b, f, e2):
        self._a = a  # equtorial radius
        self._b = b  # polar radius
        self._f = f  # flattening
        self._e2 = e2  # eccentricity squared

        # define other parameters
        self._e = np.sqrt(self._e2)  # eccentricity
        self._e12 = self._e2 / (1 - self._e2)  # 2nd eccentricity squared
        self._e1 = np.sqrt(self._e12)  # 2nd eccentricity

    #########################################################################
    # Defining and computed constants
    #########################################################################
    @property
    def equatorial_radius(self):
        """Return semi-major or equatorial axis radius"""
        return self._a

    @property
    def polar_radius(self):
        """Return semi-minor or polar axis radius"""
        return self._b

    @property
    def flattening(self):
        """Return flattening"""
        return self._f

    @property
    def reciprocal_flattening(self):
        """Return reciprocal flattening"""
        return 1 / self.flattening

    @property
    def eccentricity(self):
        """Return first eccentricity"""
        return self._e

    @property
    def eccentricity_squared(self):
        """Return first eccentricity squared"""
        return self._e2

    @property
    def second_eccentricity(self):
        """Return second eccentricity"""
        return self._e1

    @property
    def second_eccentricity_squared(self):
        """Return second eccentricity squared"""
        return self._e12

    @property
    def linear_eccentricity(self):
        """Return linear eccentricity"""
        return self.equatorial_radius * self.eccentricity

    @property
    def quadrant_distance(self):
        """Return arc of meridian from equator to pole (meridian quadrant)"""
        prc = self.polar_radius_of_curvature
        return prc * np.pi / 2 * (1 -
                                  3 / 4 * self._e12 + 45 / 64 * self._e12**2 -
                                  175 / 256 * self._e12 ** 3 +
                                  11025 / 16384 * self._e12**4)

    #########################################################################
    # Auxiliary methods
    #########################################################################
    def _w(self, lat):
        """Return auxiliary function W"""
        return np.sqrt(1 - self._e2 * np.sin(lat) ** 2)

    def _v(self, lat):
        """Return auxiliary function V"""
        return np.sqrt(1 + self._e12 * np.cos(lat) ** 2)

    #########################################################################
    # Radiuses of curvature
    #########################################################################
    @property
    def polar_radius_of_curvature(self):
        """Return polar radius of curvature"""
        return self.equatorial_radius**2 / self.polar_radius

    def curvature_radius(self, lat):
        """Return radius of curvature"""
        return self.polar_radius_of_curvature / self._v(lat) ** 3

    ##########################################################################
    def polar_equation(self, lat):
        """Return radius of the ellipse with respect to the origin

        Input:
            lat (float): spherical latitude in radians
        """
        return (self._a * self._b) / (np.sqrt(self._a**2 * np.sin(lat)**2 +
                                              self._b**2 * np.cos(lat)**2))


class Ellipsoid(_Ellipse):
    """Class represents an ellipsoid of revolution and its geometry.

    This class intialize proj.Geod class from pyproj package, so any valid init
    string for Proj are accepted as arguments. See pyproj.Geod.__new__ methods
    documentation (https://jswhit.github.io/pyproj/pyproj.Geod-class.html)
    for more information.

    Parameters
    ----------
    ellps : str, optional
        Ellipsoid name, most common ellipsoids are accepted
    """

    def __init__(self, ellps=None, **kwargs):
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

        self._geod = _proj.Geod(**kwargs)
        self._name = kwargs['ellps']

        super().__init__(self._geod.a, self._geod.b,
                         self._geod.f, self._geod.es)

    @property
    def surface_area(self):
        """Return surface area of the ellipsoid"""
        return _2pi * self._a**2 * (
            1 + 0.5 * (1 - self._e2) / self._e * np.log((1 +
                                                         self._e) / (1 - self._e)))

    @property
    def volume(self):
        """Return volume of the elliposid"""
        return _4pi * self._a**2 * self._b / 3

    ##########################################################################
    # Equivalent sphere radiuses
    ##########################################################################
    @property
    def mean_radius(self):
        """Return arithmetic mean radius"""
        return (2 * self._a + self._b) / 3

    @property
    def mean_radius_same_surface(self):
        """Return radius of the sphere with the same surface"""
        prc = self.polar_radius_of_curvature
        return prc * (1 -
                      2 / 3 * self._e12 + 26 / 45 * self._e12**2 -
                      100 / 189 * self._e12**3 +
                      7034 / 14175 * self._e12**4)

    @property
    def mean_radius_same_volume(self):
        """Return radius of the sphere with the same volume"""
        return np.power(self._a**2 * self._b, 1 / 3)

    #########################################################################
    # Radiuses of curvature
    #########################################################################
    def meridian_curvature_radius(self, lat):
        """Return radius of curvature of meridian normal section M"""
        return self.curvature_radius(lat)

    def prime_vertical_curvature_radius(self, lat):
        """Return radius of curvature of prime vertical normal section N"""
        return self.polar_radius_of_curvature / self._v(lat)

    def gaussian_curvature_radius(self, lat):
        """Return Gaussian radius of curvature"""
        meridian_curv_radius = self.meridian_curvature_radius(lat)
        pvertical_curv_radius = self.prime_vertical_curvature_radius(lat)
        return np.sqrt(meridian_curv_radius * pvertical_curv_radius)

    def mean_curvature(self, lat):
        """Return mean curvature"""
        return 0.5 * (1 / self.prime_vertical_curvature_radius(lat) +
                      1 / self.meridian_curvature_radius(lat))

    ##########################################################################
    def reduced_latitude(self, lat):
        """Return reduced latitude from geodetic one"""
        return np.arctan((1 - self._f) * np.tan(lat))

    def inverse(self):
        """Solve inverse geodetic problem on the ellipsoid"""
        raise NotImplementedError('Not implemented yet.')

    def forward(self):
        """Solve forward geodetic problem on the ellipsoid"""
        raise NotImplementedError('Not implemented yet.')

    def area(self):
        """Return area of the rectangle on the ellipsoid"""
        raise NotImplementedError('Not implemented yet.')
