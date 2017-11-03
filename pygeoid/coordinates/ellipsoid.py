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
        Ellipsoid name, most common ellipsoids are accepted
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

    #########################################################################
    # Defining and computed constants
    #########################################################################
    @property
    def equatorial_radius(self):
        """Return semi-major or equatorial axis radius"""
        return self.a

    @property
    def polar_radius(self):
        """Return semi-minor or polar axis radius"""
        return self.b

    @property
    def flattening(self):
        """Return flattening"""
        return self.f

    @property
    def reciprocal_flattening(self):
        """Return reciprocal flattening"""
        return 1 / self.flattening

    @property
    def eccentricity(self):
        """Return first eccentricity"""
        return self.e

    @property
    def eccentricity_squared(self):
        """Return first eccentricity squared"""
        return self.e2

    @property
    def second_eccentricity(self):
        """Return second eccentricity"""
        return self.e1

    @property
    def second_eccentricity_squared(self):
        """Return second eccentricity squared"""
        return self.e12

    @property
    def linear_eccentricity(self):
        """Return linear eccentricity"""
        return self.equatorial_radius * self.eccentricity

    @property
    def quadrant_distance(self):
        """Return arc of meridian from equator to pole (meridian quadrant)"""
        prc = self.polar_radius_of_curvature
        return prc * np.pi / 2 * (1 -
                                  3 / 4 * self.e12 + 45 / 64 * self.e12**2 -
                                  175 / 256 * self.e12 ** 3 +
                                  11025 / 16384 * self.e12**4)

    @property
    def surface_area(self):
        """Return surface area of the ellipsoid"""
        return _2pi * self.a**2 * (
            1 + 0.5 * (1 - self.e2) / self.e * np.log((1 +
                                                       self.e) / (1 - self.e)))

    @property
    def volume(self):
        """Return volume of the elliposid"""
        return _4pi * self.a**2 * self.b / 3

    ##########################################################################
    # Equivalent sphere radiuses
    ##########################################################################
    @property
    def mean_radius(self):
        """Return arithmetic mean radius"""
        return (2 * self.a + self.b) / 3

    @property
    def mean_radius_same_surface(self):
        """Return radius of the sphere with the same surface"""
        prc = self.polar_radius_of_curvature
        return prc * (1 -
                      2 / 3 * self.e12 + 26 / 45 * self.e12**2 -
                      100 / 189 * self.e12**3 +
                      7034 / 14175 * self.e12**4)

    @property
    def mean_radius_same_volume(self):
        """Return radius of the sphere with the same volume"""
        return np.power(self.a**2 * self.b, 1 / 3)

    #########################################################################
    # Auxiliary methods
    #########################################################################
    def _w(self, lat):
        """Return auxiliary function W"""
        return np.sqrt(1 - self.e2 * np.sin(lat) ** 2)

    def _v(self, lat):
        """Return auxiliary function V"""
        return np.sqrt(1 + self.e12 * np.cos(lat) ** 2)

    #########################################################################
    # Radiuses of curvature
    #########################################################################
    @property
    def polar_radius_of_curvature(self):
        """Return polar radius of curvature"""
        return self.equatorial_radius**2 / self.polar_radius

    def meridian_curvature_radius(self, lat):
        """Return radius of curvature of meridian normal section M"""
        return self.polar_radius_of_curvature / self._v(lat) ** 3

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
    def polar_equation(self, lat):
        """Return radius of the ellipse with respect to the origin

        Parameters
        ----------
            lat : float or array_like of floats
                Spherical latitude in radians.

        Returns
        -------
            float : radius in meters
        """
        return (self.a * self.b) / (np.sqrt(self.a**2 * np.sin(lat)**2 +
                                            self.b**2 * np.cos(lat)**2))

    def reduced_latitude(self, lat):
        """Return reduced latitude from geodetic one"""
        return np.arctan((1 - self.f) * np.tan(lat))
