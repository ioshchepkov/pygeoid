""" This module contains Position3D class.

"""

import astropy.units as u
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

    @u.quantity_input
    def __init__(self, x: u.m, y: u.m, z: u.m):
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    @u.quantity_input
    def from_geodetic(cls, lat: u.deg, lon: u.deg, height: u.m, ell):
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

        """
        x, y, z = transform.geodetic_to_cartesian(lat, lon, height, ell=ell)
        return cls(x, y, z)

    @classmethod
    @u.quantity_input
    def from_spherical(cls, lat: u.deg, lon: u.deg, radius: u.m):
        """Position, initialized from spherical coordinates.

        Parameters
        ----------
        lat, lon : float or array_like of floats
            Spherical latitude and longitude.
        radius : float or array_like of floats
            Radius, in metres.
        """
        x, y, z = transform.spherical_to_cartesian(lat, lon, radius)
        return cls(x, y, z)

    @classmethod
    @u.quantity_input
    def from_ellipsoidal(cls, rlat: u.deg, lon: u.deg, u: u.m, ell):
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
        """
        x, y, z = transform.ellipsoidal_to_cartesian(rlat, lon, u, ell=ell)
        return cls(x, y, z)

    @property
    def cartesian(self):
        """Return 3D cartesian coordinates"""
        return self._x, self._y, self._z

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

    def geodetic(self, ell):
        """Return geodetic coordinates.

        Parameters
        ----------
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates are referenced to.

        Returns
        -------
        lat, lon : float or array_like of floats
            Geodetic latitude and longitude.
        height : float or array_like of floats
            Geodetic height, in metres.
        """
        lat, lon, height = transform.cartesian_to_geodetic(
            self._x, self._y, self._z, ell=ell)
        return lat, lon, height

    def spherical(self):
        """Return spherical (geocentric) coordinates.

        Returns
        -------
        lat, lon : float or array_like of floats
            Spherical latitude and longitude.
        r : float or array_like of floats
            Radius, in metres.
        """
        lat, lon, radius = transform.cartesian_to_spherical(
            self._x, self._y, self._z)
        return lat, lon, radius

    def ellipsoidal(self, ell):
        """Return ellipsoidal-harmonic coordinates.

        Parameters
        ----------
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which ellipsoidal coordinates are referenced to.

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
            self._x, self._y, self._z, ell=ell)
        return rlat, lon, u

    @u.quantity_input
    def enu(self, origin: u.deg, ell=None):
        """Return local east-north-up cartesian coordinates.

        Parameters
        ----------
        origin : array_like of floats
            Ggeocentric (spherical) or geodetic coordinates of the origin
            (`lat0`, `lon0`, `r0`) or (`lat0`, `lon0`, `h0`).
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates
            are referenced to. Default is None, meaning spherical
            coordinates instead of geodetic.

        Returns
        -------
        x, y, z : float or array_like of floats
            Local east-north-up cartesian coordinates, in metres.
        """
        east, north, up = transform.ecef_to_enu(
            self._x, self._y, self._z, origin, ell=ell)

        return east, north, up
