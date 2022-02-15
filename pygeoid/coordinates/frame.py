""" This module contains frame classes.

"""

import astropy.units as u
import numpy as _np
from astropy.coordinates import (Attribute, BaseCoordinateFrame,
                                 FunctionTransform, TimeAttribute,
                                 frame_transform_graph)
from astropy.coordinates.angles import Latitude, Longitude
from astropy.coordinates.representation import CartesianRepresentation
from pygeoid.coordinates import transform
from pygeoid.coordinates.ellipsoid import Ellipsoid
from pygeoid.coordinates.representation import (
    EllipsoidalHarmonicRepresentation, GeodeticRepresentation)

__all__ = ["LocalFrame", "ECEF", "LocalTangentPlane"]


class LocalFrame(BaseCoordinateFrame):
    """Arbitrary local cartesian frame.

    """
    default_representation = CartesianRepresentation


class ECEF(BaseCoordinateFrame):
    """Earth-Centered, Earth-Fixed frame.

    Parameters
    ----------
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.
    *args
        Any representation of the frame data, e.g. x, y, and z coordinates.
    **kwargs
        Any extra BaseCoordinateFrame arguments.

    """

    default_representation = CartesianRepresentation
    """Default representation of local frames"""

    _ellipsoid = Ellipsoid()

    def __init__(self, *args, ell=None, **kwargs):

        super().__init__(*args, **kwargs)

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


class LocalTangentPlane(BaseCoordinateFrame):
    """Local tangent plane geodetic coordiante frame.

    Parameters
    ----------
    *args
        Any representation of the frame data, e.g. x, y, and z coordinates
    origin : `pygeoid.coordinates.frame.ECEF`
        The location on Earth of the local frame origin
    orientation : sequence of str, optional
        The cardinal directions of the x, y, and z axis (default: E, N, U)
    **kwargs
        Any extra BaseCoordinateFrame arguments

    Raises
    ------
    ValueError
        The local frame configuration is not valid

    """

    default_representation = CartesianRepresentation

    origin = Attribute()
    """The origin on Earth of the local frame"""

    orientation = Attribute(default=("E", "N", "U"))
    """The orientation of the local frame, as cardinal directions"""

    def __init__(self, *args, origin, orientation=None, **kwargs):

        super().__init__(*args, origin=origin,
                         orientation=orientation, **kwargs)

        def vector(lat, lon, name):
            _name = name[0].upper()

            azalt = {
                "E" : (90, 0),
                "W" : (270, 0),
                "N" : (0, 0),
                "S" : (180, 0),
                "U" : (0, 90),
                "D" : (0, -90)
            }

            if _name not in azalt:
                raise ValueError(f"Invalid frame orientation `{name}`")

            az, alt = azalt[_name]
            az *= u.deg
            alt *= u.deg

            calt = _np.cos(alt)
            r = [calt * _np.sin(az), calt * _np.cos(az), _np.sin(alt)]
            east, north, up = transform._ecef_to_enu_rotation_matrix(lat, lon)

            d0 = r[0] * east[0] + r[1] * north[0] + r[2] * up[0]
            d1 = r[0] * east[1] + r[1] * north[1] + r[2] * up[1]
            d2 = r[0] * east[2] + r[1] * north[2] + r[2] * up[2]

            return d0, d1, d2

        geodetic = self._origin.geodetic

        ux = vector(geodetic.lat, geodetic.lon, self._orientation[0])
        uy = vector(geodetic.lat, geodetic.lon, self._orientation[1])
        uz = vector(geodetic.lat, geodetic.lon, self._orientation[2])

        self._basis = _np.column_stack((ux, uy, uz))


@frame_transform_graph.transform(FunctionTransform,
                                 ECEF, LocalTangentPlane)
def ecef_to_local(ecef, local):
    """Compute the transformation from ECEF to LocalTangentPlane coordinates.

    Parameters
    ----------
    ecef : ECEF
        The initial coordinates in ECEF
    local : LocalTangentPlane
        The LocalTangentPlane frame to transform to

    Returns
    -------
    LocalTangentPlane
        The LocalTangentPlane frame with transformed coordinates
    """
    c = ecef.represent_as('cartesian')
    if c.x.unit.is_equivalent("m"):
        c = c.copy()
        c -= local._origin.represent_as('cartesian')

    c = c.transform(local._basis.T)

    return local.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform,
                                 LocalTangentPlane, ECEF)
def local_to_ecef(local, ecef):
    """Compute the transformation from LocalTangentPlane to ECEF coordinates.

    Parameters
    ----------
    local : LocalTangentPlane
        The initial coordinates in LocalTangentPlane
    ecef : ECEF
        The ECEF frame to transform to

    Returns
    -------
    ECEF
        The ECEF frame with transformed coordinates
    """
    c = local.represent_as('cartesian').transform(local._basis)
    if c.x.unit.is_equivalent("m"):
        c += local._origin.represent_as('cartesian')

    return ecef.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform,
                                 LocalTangentPlane, LocalTangentPlane)
def local_to_local(local0, local1):
    """Compute the transformation between LocalTangentPlane coordinates.

    Parameters
    ----------
    local0 : LocalTangentPlane
        The initial coordinates in the 1st LocalTangentPlane frame.
    local1 : LocalTangentPlane
        The 2nd LocalTangentPlane frame to transform to.

    Returns
    -------
    LocalTangentPlane
        The LocalTangentPlane frame with transformed coordinates.

    """
    c = local0.represent_as('cartesian')
    translate = c.x.unit.is_equivalent("m")

    # Check if the two frames are identicals
    if _np.array_equal(local0._basis, local1._basis):
        local0_c = local0._origin.represent_as('cartesian')
        local1_c = local1._origin.represent_as('cartesian')
        if not translate or ((local0_c.x == local1_c.x) and
                             (local0_c.y == local1_c.y) and
                             (local0_c.z == local1_c.z)):
            # CartesianRepresentations might not eveluate to equal though the
            # coordinates are equal
            return local1.realize_frame(c)

    # Transform from Local0 to ECEF
    c = c.transform(local0._basis)
    if translate:
        c = c.copy()
        c += local0._origin.represent_as('cartesian')

    # Transform back from ECEF to Local1
    if translate:
        c = c.copy()
        c -= local1._origin.represent_as('cartesian')

    c = c.transform(local1._basis.T)

    return local1.realize_frame(c)
