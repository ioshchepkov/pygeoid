""" This module contains Position3D class.

"""

from pygeoid.coordinates import transform


class Position3D:
    """Class represents a 3D positioning vector.

    The main purpose of the class is to store 3D positioning of the points
    in cartesian coordinates, so they can be easily transformed to
    curvilinear coordinates: geodetic, spherical, elliposidal-harmonic.

    Parameters
    ----------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
    """

    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_geodetic(cls, lat, lon, height, ell, degrees=True):
        """Position, initialized from geodetic coordinates.

        Parameters
        ----------
        lat : float or array_like of floats
            Geodetic latitude.
        lon : float or array_like of floats
            Geodetic longitude.
        height : float or array_like of floats
            Geodetic height, in metres.
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates are referenced to.
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.
        """
        x, y, z = transform.geodetic_to_cartesian(lat, lon, height, ell=ell,
                                                  degrees=degrees)
        return cls(x, y, z)

    @classmethod
    def from_spherical(cls, lat, lon, radius, degrees=True):
        """Position, initialized from spherical coordinates.

        Parameters
        ----------
        lat, lon : float or array_like of floats
            Spherical latitude and longitude.
        radius : float or array_like of floats
            Radius, in metres.
        degrees : bool, optional
            If True, the input `lat` and `lon` are given in degrees,
            otherwise radians.
        """
        x, y, z = transform.spherical_to_cartesian(lat, lon, radius,
                                                   degrees=degrees)
        return cls(x, y, z)

    @classmethod
    def from_ellipsoidal(cls, rlat, lon, u, ell, degrees=True):
        """Position, initialized from ellipsoidal coordinates.

        Parameters
        ----------
        rlat : float or array_like of floats
            Reduced latitude.
        lon : float or array_like of floats
            Longitude.
        u : float or array_like of floats
            Polar axis of the ellipsoid passing through the point.
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates are referenced to.
        degrees : bool, optional
            If True, the input `rlat` and `lon` are given in degrees,
            otherwise radians.
        """
        x, y, z = transform.ellipsoidal_to_cartesian(rlat, lon, u, ell=ell,
                                                     degrees=degrees)
        return cls(x, y, z)

    @property
    def cartesian(self):
        """Return 3D cartesian coordinates"""
        return (self._x, self._y, self._z)

    @property
    def x(self):
        """Return x coordinate"""
        return self._x

    @property
    def y(self):
        """Return y coordinate"""
        return self._y

    @property
    def z(self):
        """Return z coordinate"""
        return self._z

    def geodetic(self, ell, degrees=True):
        """Return geodetic coordinates.

        Parameters
        ----------
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates are referenced to.
        degrees : bool, optional
            If True, the output geodetic latitude and longitude will be in degrees,
            otherwise radians.

        Returns
        -------
        lat, lon : float or array_like of floats
            Geodetic latitude and longitude.
        height : float or array_like of floats
            Geodetic height, in metres.
        """
        lat, lon, height = transform.cartesian_to_geodetic(
            self._x, self._y, self._z, ell=ell, degrees=degrees)
        return lat, lon, height

    def spherical(self, degrees=True):
        """Return spherical (geocentric) coordinates.

        Parameters
        ----------
        degrees : bool, optional
            If True, the output spherical latitude and longitude will be in degrees,
            otherwise radians.

        Returns
        -------
        lat, lon : float or array_like of floats
            Spherical latitude and longitude.
        r : float or array_like of floats
            Radius, in metres.
        """
        lat, lon, radius = transform.cartesian_to_spherical(
            self._x, self._y, self._z, degrees=degrees)
        return lat, lon, radius

    def ellipsoidal(self, ell, degrees=True):
        """Return ellipsoidal-harmonic coordinates.

        Parameters
        ----------
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which ellipsoidal coordinates are referenced to.
        degrees : bool, optional
            If True, the output reduced latitude and longitude will
            be in degrees, otherwise radians.

        Returns
        -------
        rlat : float or array_like of floats
            Reduced latitude.
        lon : float or array_like of floats
            Longitude.
        u : float or array_like of floats
            Polar axis of the ellipsoid passing through the given point.
        """
        rlat, lon, u = transform.cartesian_to_ellipsoidal(
            self._x, self._y, self._z, ell=ell, degrees=degrees)
        return rlat, lon, u
