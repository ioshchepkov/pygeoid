""" This module contains functions for coordinate transformations

"""

import functools as _functools
import numpy as _np


##############################################################################
# 3D coordinates
##############################################################################

def geodetic_to_cartesian(lat, lon, height, ell, degrees=True):
    """Convert geodetic to 3D cartesian coordinates.

    Convert geodetic coordinates (`lat`, `lon`, `height`) given w.r.t.
    ellipsoid `ell` to 3D cartesian coordinates (`x`, `y`, `z`).

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

    Returns
    -------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
    """

    if degrees:
        lat = _np.radians(lat)
        lon = _np.radians(lon)

    N = ell.prime_vertical_curvature_radius(lat)
    x = (N + height) * _np.cos(lat) * _np.cos(lon)
    y = (N + height) * _np.cos(lat) * _np.sin(lon)
    z = (N + height - N * ell.e2) * _np.sin(lat)

    return x, y, z


@_functools.partial(_np.vectorize, otypes=(_np.float64, _np.float64,
                                           _np.float64), excluded=[3, 4])
def cartesian_to_geodetic(x, y, z, ell, degrees=True):
    """Convert 3D cartesian to geodetic coordinates.

    Parameters
    ----------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
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

    Notes
    -----
    The algorithm of H. Vermeille is used for this transformation [1]_.

    References
    ----------
    .. [1] Vermeille, H., 2011. An analytical method to transform geocentric into
    geodetic coordinates. Journal of Geodesy, 85(2), pp.105-117.
    """
    e4 = ell.e2 ** 2

    # Step 1
    p = (x**2 + y**2) / ell.a ** 2
    q = (1 - ell.e2) * z ** 2 / ell.a ** 2
    r = (p + q - e4) / 6

    # Step 2 - 3
    e4pq = e4 * p * q
    t = 8 * r**3 + e4pq

    if (t > 0) or (t <= 0 and q != 0):
        if t > 0:
            l = _np.power(_np.sqrt(t) + _np.sqrt(e4pq), 1 / 3)
            u = 3 / 2 * r**2 / l**2 + 0.5 * (l + r / l)**2
        elif t <= 0 and q != 0:
            u_aux = 2 / 3 * _np.arctan2(_np.sqrt(e4pq), _np.sqrt(-t) +
                                        _np.sqrt(-8 * r**3))

            u = -4 * r * _np.sin(u_aux) * _np.cos(_np.pi / 6 + u_aux)

        v = _np.sqrt(u**2 + e4 * q)
        w = ell.e2 * (u + v - q) / (2 * v)
        k = (u + v) / (_np.sqrt(w**2 + u + v) + w)
        D = (k * _np.sqrt(x**2 + y**2)) / (k + ell.e2)

        height = (k + ell.e2 - 1) * _np.sqrt(D**2 + z**2) / k
        lat = 2 * _np.arctan2(z, D + _np.sqrt(D**2 + z**2))
    # Step 4
    elif q == 0 and p <= e4:
        e2p = _np.sqrt(ell.e2 - p)
        me2 = _np.sqrt(1 - ell.e2)

        height = - (ell.a * me2 * e2p) / _np.sqrt(ell.e2)
        lat = 2 * _np.arctan2(_np.sqrt(e4 - p), _np.sqrt(ell.e2) * e2p +
                              me2 * _np.sqrt(p))

    lon = _np.arctan2(y, x)

    if degrees:
        lat = _np.degrees(lat)
        lon = _np.degrees(lon)

    return lat, lon, height


def cartesian_to_spherical(x, y, z, degrees=True):
    """Convert 3D cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
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

    radius = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = _np.arctan2(z, _np.sqrt(x ** 2 + y ** 2))
    lon = _np.arctan2(y, x)

    if degrees:
        lat = _np.degrees(lat)
        lon = _np.degrees(lon)

    return lat, lon, radius


def spherical_to_cartesian(lat, lon, radius, degrees=True):
    """Convert spherical to 3D cartesian coordinates.

    Parameters
    ----------
    lat, lon : float or array_like of floats
        Spherical latitude and longitude.
    radius : float or array_like of floats
        Radius, in metres.
    degrees : bool, optional
        If True, the input `lat` and `lon` are given in degrees,
        otherwise radians.

    Returns
    -------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
    """

    if degrees:
        lat = _np.radians(lat)
        lon = _np.radians(lon)

    x = radius * _np.cos(lat) * _np.cos(lon)
    y = radius * _np.cos(lat) * _np.sin(lon)
    z = radius * _np.sin(lat)

    return x, y, z


def cartesian_to_ellipsoidal(x, y, z, ell, degrees=True):
    """Convert 3D cartesian to ellipsoidal-harmonic coordinates.

    Note that point (x, y, z) must be on or outside of the sphere with the
    radius equals to the linear eccentricity of the reference ellipsoid `ell`,
    i. e. (x**2 + y**2 + z**2) >= E**2.

    Parameters
    ----------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
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

    le2 = ell.linear_eccentricity**2

    k = x**2 + y**2 + z**2 - le2

    if _np.any(k < 0):
        raise ValueError(
            'x**2 + y**2 + z**2 must be grater or equal to ' +
            'the linear eccentricity of the reference ellipsoid.')

    u = k * (0.5 + 0.5 * _np.sqrt(1 + (4 * le2 * z**2) / k**2))

    u = _np.sqrt(u)
    rlat = _np.arctan2(z * _np.sqrt(u ** 2 + le2),
                       u * _np.sqrt(x**2 + y**2))

    lon = _np.arctan2(y, x)

    if degrees:
        rlat = _np.degrees(rlat)
        lon = _np.degrees(lon)

    return rlat, lon, u


def ellipsoidal_to_cartesian(rlat, lon, u, ell, degrees=True):
    """Convert ellipsoidal-harmonic coordinates to 3D cartesian coordinates.

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

    Returns
    -------
    x, y, z : float or array_like of floats
        Cartesian coordinates, in metres.
    """

    if degrees:
        rlat = _np.radians(rlat)
        lon = _np.radians(lon)

    k = _np.sqrt(u**2 + ell.linear_eccentricity**2)

    x = k * _np.cos(rlat) * _np.cos(lon)
    y = k * _np.cos(rlat) * _np.sin(lon)
    z = u * _np.sin(rlat)

    return x, y, z


def geodetic_to_spherical(lat, lon, height, ell, degrees=True):
    """Convert from geodetic to spherical coordinates.

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
        If True, the input and output `lat` and `lon` are given in degrees,
        otherwise radians.

    Returns
    -------
    lat, lon : float or array_like of floats
        Spherical latitude and longitude.
    r : float or array_like of floats
        Radius, in metres.
    """
    return cartesian_to_spherical(
        *geodetic_to_cartesian(lat, lon, height, ell=ell, degrees=degrees),
        degrees=degrees)


def spherical_to_geodetic(lat, lon, radius, ell, degrees=True):
    """Convert spherical to geodetic coordinates.

    Parameters
    ----------
    lat, lon : float or array_like of floats
        Spherical latitude and longitude.
    radius : float or array_like of floats
        Radius, in metres.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.
    degrees : bool, optional
        If True, the input and output `lat` and `lon` are given in degrees,
        otherwise radians.

    Returns
    -------
    lat, lon : float or array_like of floats
        Geodetic latitude and lonfitude.
    height : float or array_like of floats
        Geodetic height, in metres.
    """
    return cartesian_to_geodetic(
        *spherical_to_cartesian(lat, lon, radius, degrees=degrees),
        ell=ell, degrees=degrees)


def geodetic_to_ellipsoidal(lat, lon, height, ell, degrees=True):
    """Convert from geodetic to ellipsoidal-harmonic coordinates.

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
        If True, the input `lat`, `lon` are and the output `rlat`,`lon`
        will be given in degrees, otherwise radians.

    Returns
    -------
    rlat : float or array_like of floats
        Reduced latitude.
    lon : float or array_like of floats
        Longitude.
    u : float or array_like of floats
        Polar axis of the ellipsoid passing through
        the given point.
    """
    return cartesian_to_ellipsoidal(
        *geodetic_to_cartesian(lat, lon, height, ell=ell, degrees=degrees),
        ell=ell, degrees=degrees)


def ellipsoidal_to_geodetic(rlat, lon, u, ell, degrees=True):
    """Convert from ellipsoidal-harmonic to geodetic coordinates.

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

    Returns
    -------
    lat, lon : float or array_like of floats
        Geodetic latitude and longitude.
    height : float or array_like of floats
        Geodetic height, in metres.
    """
    return cartesian_to_geodetic(
        *ellipsoidal_to_cartesian(rlat, lon, u, ell=ell, degrees=degrees),
        ell=ell, degrees=degrees)

##############################################################################
# 2D coordinates
##############################################################################


def polar_to_cartesian(theta, radius, degrees=True):
    """Convert polar coordinates to 2D cartesian.

    Parameters
    ----------
    theta : float or array_like of floats
        Polar angle.
    radius : float or array_like of floats
        Radius, in metres.

    Returns
    -------
    x, y : float or array_like of floats
        Cartesian coordinates, in metres.
    """
    if degrees:
        theta = _np.radians(theta)

    return radius * _np.cos(theta), radius * _np.sin(theta)


def cartesian_to_polar(x, y, degrees=True):
    """Convert 2D cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x, y : float or array_like of floats
        Cartesian coordinates, in metres.
    degrees : bool, optional
        If True, the output azimuth will be in degrees, otherwise radians.

    Returns
    -------
    theta : float or array_like of floats
        Polar angle.
    radius : float or array_like of floats
        Radius, in metres.
    """

    radius = _np.sqrt(x ** 2 + y ** 2)
    theta = _np.arctan2(y, x)

    if degrees:
        theta = _np.degrees(theta)

    return theta, radius
