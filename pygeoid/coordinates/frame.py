""" This module contains ECEF class.

"""

import astropy.units as u

from astropy.coordinates.angles import Longitude, Latitude
from astropy.coordinates import BaseCoordinateFrame
from astropy.coordinates import TimeAttribute

from astropy.coordinates.representation import CartesianRepresentation
from pygeoid.coordinates.representation import (GeodeticRepresentation,
                                                EllipsoidalHarmonicRepresentation)

from pygeoid.coordinates import transform
from pygeoid.coordinates.ellipsoid import Ellipsoid


class ECEF(BaseCoordinateFrame):
    """Earth-Centered, Earth-Fixed frame.

    Parameters
    ----------
    obstime : `~astropy.time.Time` or datetime or str, optional
        The observation time
    *args
        Any representation of the frame data, e.g. x, y, and z coordinates.
    **kwargs
        Any extra BaseCoordinateFrame arguments.

    """

    default_representation = CartesianRepresentation
    obstime = TimeAttribute(default=None)

    _ellipsoid = Ellipsoid()

    def __init__(self, *args, obstime=None, ell=None, **kwargs):

        super().__init__(*args, obstime=obstime, **kwargs)

        if ell is not None:
            self._ellipsoid = ell

    @property
    def ellipsoid(self):
        """Reference ellipsoid.

        """
        return self._ellipsoid

    @ellipsoid.setter
    def ellipsoid(self, ellipsoid):
        if not isinstance(ellipsoid, Ellipsoid):
            raise ValueError('elliposid should be an instance of the '
                             '`pygeoid.coordinates.ellipsoid.Ellipsoid`!'
                             )
        else:
            self._ellipsoid = ellipsoid

    @property
    def cartesian(self):
        return (self.x, self.y, self.z)

    @classmethod
    def from_spherical(cls, lat, lon, radius):
        """Position, initialized from spherical coordinates.

        Parameters
        ----------
        lat : ~astropy.units.Quantity or array-like
            Spherical latitude. Can be anything that initialises an
            `~astropy.coordinates.Latitude` object.
            (if array-like, in degrees).
        lon : ~astropy.units.Quantity or array-like
            Spherical longitude. Can be anything that initialises an
            `~astropy.coordinates.Longitude` object.
            (if array-like, in degrees).
        radius : ~astropy.units.Quantity or array-like
            Radius (if array-like, in metres).
        """
        lat = Latitude(lat, u.degree, copy=False)
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)

        if not isinstance(radius, u.Quantity):
            radius = u.Quantity(radius, u.m, copy=False)

        x, y, z = u.Quantity(
            transform.spherical_to_cartesian(lat, lon, radius))

        return cls(x, y, z)

    @classmethod
    def from_geodetic(cls, lat, lon, height=0., ell=None):
        """Position, initialized from geodetic coordinates.

        Parameters
        ----------
        lat : ~astropy.units.Quantity or array-like
            Geodetic latitude. Can be anything that initialises an
            `~astropy.coordinates.Latitude` object (if array-like, in degrees).
        lon : ~astropy.units.Quantity or array-like
            Geodetic longitude. Can be anything that initialises an
            `~astropy.coordinates.Longitude` object (if array-like, in degrees).
        height : ~astropy.units.Quantity or array-like
            Geodetic height (if array-like, in metres). Default is 0 m.
        ell : ~`pygeoid.coordinates.ellipsoid.Ellipsoid`, optional
            Reference ellipsoid to which geodetic coordinates are referenced to.
            Default is None, which means the default ellipsoid of the class
            instance, but if given, it also will change the ellipsoid for
            the class instance.

        """
        lat = Latitude(lat, u.degree, copy=False)
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)

        if not isinstance(height, u.Quantity):
            height = u.Quantity(height, u.m, copy=False)

        if ell is None:
            ell = cls._ellipsoid

        x, y, z = u.Quantity(
            transform.geodetic_to_cartesian(lat, lon, height, ell))

        self = cls(x, y, z)
        self._ellipsoid = ell

        return self

    @property
    def geodetic(self):
        return self.represent_as('geodetic', in_frame_units=True)

    @classmethod
    def from_ellipsoidal_harmonic(cls, rlat, lon, u_ax, ell=None):
        """Position, initialized from ellipsoidal-harmonic coordinates.

        Parameters
        ----------
        rlat : ~astropy.units.Quantity or array-like
            Reduced latitude. Can be anything that initialises an
            `~astropy.coordinates.Latitude` object.
        lon : ~astropy.units.Quantity or array-like
            Spherical longitude. Can be anything that initialises an
            `~astropy.coordinates.Longitude` object.
            (if array-like, in degrees).
        u_ax : ~astropy.units.Quantity or array-like
            Polar axis of the ellipsoid passing through the given point
            (if array-like, in metres).
        ell : ~`pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which coordinates are referenced to.
            Default is None, which means the default ellipsoid of the class
            instance, but if given, it also will change the ellipsoid for
            the class instance.
        """
        rlat = Latitude(rlat, u.degree, copy=False)
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)

        if not isinstance(u_ax, u.Quantity):
            u_ax = u.Quantity(u_ax, u.m, copy=False)

        if ell is not None:
            ell = cls._ellipsoid

        x, y, z = u.Quantity(
            transform.ellipsoidal_to_cartesian(rlat, lon, u_ax, ell=ell))

        self = cls(x, y, z)
        self._ellipsoid = ell

        return self

    @property
    def ellipsoidal_harmonic(self):
        return self.represent_as('ellipsoidalharmonic', in_frame_units=True)

    @u.quantity_input
    def enu(self, origin: tuple[u.deg, u.deg, u.m], ell=None):
        """Return local east-north-up cartesian coordinates.

        Parameters
        ----------
        origin : tuple of ~astropy.units.Quantity
            Ggeocentric (spherical) or geodetic coordinates of the origin
            (`lat0`, `lon0`, `r0`) or (`lat0`, `lon0`, `h0`).
        ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
            Reference ellipsoid to which geodetic coordinates
            are referenced to. Default is None, meaning spherical
            coordinates instead of geodetic.

        Returns
        -------
        east, north, up : ~astropy.units.Quantity
            Local east-north-up cartesian coordinates.
        """
        if ell is not None:
            ell = self._ellipsoid

        east, north, up = transform.ecef_to_enu(
            self.x, self.y, self.z, origin, ell=ell)

        return east, north, up
