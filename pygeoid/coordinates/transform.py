""" This module contains functions for coordinate transformations

"""

import functools as _functools
import numpy as _np
import astropy.units as u


##############################################################################
# 3D coordinates
##############################################################################

@u.quantity_input
def geodetic_to_cartesian(
        lat: u.deg, lon: u.deg, height: u.m, ell):
    """Convert geodetic to 3D cartesian coordinates.

    Convert geodetic coordinates (`lat`, `lon`, `height`) given w.r.t.
    ellipsoid `ell` to 3D cartesian coordinates (`x`, `y`, `z`).

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    x, y, z : ~astropy.units.Quantity
        Cartesian coordinates.
    """

    N = ell.prime_vertical_curvature_radius(lat)
    x = (N + height) * _np.cos(lat) * _np.cos(lon)
    y = (N + height) * _np.cos(lat) * _np.sin(lon)
    z = (N + height - N * ell.e2) * _np.sin(lat)

    return x, y, z


@_functools.partial(_np.vectorize,
                    otypes=(_np.float64, _np.float64, _np.float64), excluded=[3, 4])
def _cartesian_to_geodetic(x, y, z, ell, degrees=True):
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
    .. [1] Vermeille, H., 2011. An analytical method to transform geocentric
    into geodetic coordinates. Journal of Geodesy, 85(2), pp.105-117.
    """
    e2 = ell.e2.value
    a = ell.a.value

    e4 = e2 ** 2

    # Step 1
    p = (x**2 + y**2) / a ** 2
    q = (1 - e2) * z ** 2 / a ** 2
    r = (p + q - e4) / 6

    # Step 2 - 3
    e4pq = e4 * p * q
    t = 8 * r**3 + e4pq

    if (t > 0) or (t <= 0 and q != 0):
        if t > 0:
            li = _np.power(_np.sqrt(t) + _np.sqrt(e4pq), 1 / 3)
            u = 3 / 2 * r**2 / li**2 + 0.5 * (li + r / li)**2
        elif t <= 0 and q != 0:
            u_aux = 2 / 3 * _np.arctan2(_np.sqrt(e4pq), _np.sqrt(-t) +
                                        _np.sqrt(-8 * r**3))

            u = -4 * r * _np.sin(u_aux) * _np.cos(_np.pi / 6 + u_aux)

        v = _np.sqrt(u**2 + e4 * q)
        w = e2 * (u + v - q) / (2 * v)
        k = (u + v) / (_np.sqrt(w**2 + u + v) + w)
        D = (k * _np.sqrt(x**2 + y**2)) / (k + e2)

        height = (k + e2 - 1) * _np.sqrt(D**2 + z**2) / k
        lat = 2 * _np.arctan2(z, D + _np.sqrt(D**2 + z**2))
    # Step 4
    elif q == 0 and p <= e4:
        e2p = _np.sqrt(e2 - p)
        me2 = _np.sqrt(1 - e2)

        height = - (a * me2 * e2p) / _np.sqrt(e2)
        lat = 2 * _np.arctan2(_np.sqrt(e4 - p), _np.sqrt(e2) * e2p +
                              me2 * _np.sqrt(p))

    lon = _np.arctan2(y, x)

    if degrees:
        lat = _np.degrees(lat)
        lon = _np.degrees(lon)

    return lat, lon, height


@u.quantity_input
def cartesian_to_geodetic(x: u.m, y: u.m, z: u.m, ell):
    """Convert 3D cartesian to geodetic coordinates.

    Parameters
    ----------
    x, y, z : ~astropy.units.Quantity
        Cartesian coordinates.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : float or array_like of floats
        Geodetic height.

    Notes
    -----
    The algorithm of H. Vermeille is used for this transformation [1]_.

    References
    ----------
    .. [1] Vermeille, H., 2011. An analytical method to transform geocentric
    into geodetic coordinates. Journal of Geodesy, 85(2), pp.105-117.
    """

    lat, lon, height = _cartesian_to_geodetic(
        x.to('m').value, y.to('m').value, z.to('m').value, ell=ell,
        degrees=True)

    return lat * u.deg, lon * u.deg, height * u.m


@u.quantity_input
def cartesian_to_spherical(x: u.m, y: u.m, z: u.m):
    """Convert 3D cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : ~astropy.units.Quantity
        Cartesian coordinates.

    Returns
    -------
    lat : ~astropy.units.Quantity
        Spherical latitude.
    lon : ~astropy.units.Quantity
        Spherical longitude.
    r : ~astropy.units.Quantity
        Radius.
    """

    radius = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = _np.arctan2(z, _np.sqrt(x ** 2 + y ** 2))
    lon = _np.arctan2(y, x)

    return lat, lon, radius


@u.quantity_input
def spherical_to_cartesian(lat: u.deg, lon: u.deg, radius: u.m):
    """Convert spherical to 3D cartesian coordinates.

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Spherical latitude.
    lon : ~astropy.units.Quantity
        Spherical longitude.
    r : ~astropy.units.Quantity
        Radius.

    Returns
    -------
    x, y, z : ~astropy.units.Quantity
        Cartesian coordinates.
    """

    x = radius * _np.cos(lat) * _np.cos(lon)
    y = radius * _np.cos(lat) * _np.sin(lon)
    z = radius * _np.sin(lat)

    return x, y, z


@u.quantity_input
def cartesian_to_ellipsoidal(x: u.m, y: u.m, z: u.m, ell):
    """Convert 3D cartesian to ellipsoidal-harmonic coordinates.

    Note that point (x, y, z) must be on or outside of the sphere with the
    radius equals to the linear eccentricity of the reference ellipsoid `ell`,
    i. e. (x**2 + y**2 + z**2) >= E**2.

    Parameters
    ----------
    x, y, z : ~astropy.units.Quantity
        Cartesian coordinates.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which ellipsoidal coordinates are referenced to.

    Returns
    -------
    rlat : ~astropy.units.Quantity
        Reduced latitude.
    lon : ~astropy.units.Quantity
        Longitude.
    u_ax : ~astropy.units.Quantity
        Polar axis of the ellipsoid passing through the given point.
    """

    le2 = ell.linear_eccentricity**2

    k = x**2 + y**2 + z**2 - le2

    if _np.any(k < 0):
        raise ValueError(
            'x**2 + y**2 + z**2 must be grater or equal to ' +
            'the linear eccentricity of the reference ellipsoid.')

    u_ax = k * (0.5 + 0.5 * _np.sqrt(1 + (4 * le2 * z**2) / k**2))

    u_ax = _np.sqrt(u_ax)
    rlat = _np.arctan2(z * _np.sqrt(u_ax ** 2 + le2),
                       u_ax * _np.sqrt(x**2 + y**2))

    lon = _np.arctan2(y, x)

    return rlat, lon, u_ax


@u.quantity_input
def ellipsoidal_to_cartesian(rlat: u.deg, lon: u.deg, u_ax: u.m, ell):
    """Convert ellipsoidal-harmonic coordinates to 3D cartesian coordinates.

    Parameters
    ----------
    rlat : ~astropy.units.Quantity
        Reduced latitude.
    lon : ~astropy.units.Quantity
        Longitude.
    u_ax : ~astropy.units.Quantity
        Polar axis of the ellipsoid passing through the given point.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    x, y, z : ~astropy.units.Quantity
        Cartesian coordinates.
    """

    k = _np.sqrt(u_ax**2 + ell.linear_eccentricity**2)

    x = k * _np.cos(rlat) * _np.cos(lon)
    y = k * _np.cos(rlat) * _np.sin(lon)
    z = u_ax * _np.sin(rlat)

    return x, y, z


@u.quantity_input
def geodetic_to_spherical(lat: u.deg, lon: u.deg, height: u.m, ell):
    """Convert from geodetic to spherical coordinates.

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    lat, lon : ~astropy.units.Quantity
        Spherical latitude and longitude.
    r : ~astropy.units.Quantity
        Radius.
    """
    return cartesian_to_spherical(
        *geodetic_to_cartesian(lat, lon, height, ell=ell))


@u.quantity_input
def spherical_to_geodetic(lat: u.deg, lon: u.deg, radius: u.m, ell):
    """Convert spherical to geodetic coordinates.

    Parameters
    ----------
    lat, lon : ~astropy.units.Quantity
        Spherical latitude and longitude.
    r : ~astropy.units.Quantity
        Radius.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    """
    return cartesian_to_geodetic(
        *spherical_to_cartesian(lat, lon, radius), ell=ell)


@u.quantity_input
def geodetic_to_ellipsoidal(lat: u.deg, lon: u.deg, height: u.m, ell):
    """Convert from geodetic to ellipsoidal-harmonic coordinates.

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    rlat : ~astropy.units.Quantity
        Reduced latitude.
    lon : ~astropy.units.Quantity
        Longitude.
    u_ax : ~astropy.units.Quantity
        Polar axis of the ellipsoid passing through the given point.

    """
    return cartesian_to_ellipsoidal(
        *geodetic_to_cartesian(lat, lon, height, ell=ell), ell=ell)


@u.quantity_input
def ellipsoidal_to_geodetic(rlat: u.deg, lon: u.deg, u_ax: u.m, ell):
    """Convert from ellipsoidal-harmonic to geodetic coordinates.

    Parameters
    ----------
    rlat : ~astropy.units.Quantity
        Reduced latitude.
    lon : ~astropy.units.Quantity
        Longitude.
    u_ax : ~astropy.units.Quantity
        Polar axis of the ellipsoid passing through the given point.
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    lat, lon : ~astropy.units.Quantity
        Geodetic latitude and longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    """
    return cartesian_to_geodetic(
        *ellipsoidal_to_cartesian(rlat, lon, u_ax, ell=ell), ell=ell)


@u.quantity_input
def _ecef_to_enu_rotation_matrix(lat: u.deg, lon: u.deg):
    """Return ECEF to ENU rotation matrix.

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Geodetic or spherical latitude.
    lon : ~astropy.units.Quantity
        Geodetic or spherical longitude.

    """
    clat = _np.cos(lat)
    slat = _np.sin(lat)
    clon = _np.cos(lon)
    slon = _np.sin(lon)

    rotation_matrix = _np.array([
        [-slon, clon, 0],
        [-slat * clon, -slat * slon, clat],
        [clat * clon, clat * slon, slat]])

    return rotation_matrix


@u.quantity_input
def ecef_to_enu(x: u.m, y: u.m, z: u.m, origin: tuple[u.deg, u.deg, u.m], ell=None):
    """Convert geocentric cartesian to local cartesian coordinates.

    Convert Earth Centered Earth Fixed (ECEF) cartesian coordinates
    (`x`,`y`,`z`) to the local east-north-up (ENU) local cartesian
    coordinate system with the origin in (`lat0`, `lon0`, `height0`)
    or in (`lat0`, `lon0`, `r0`).

    Parameters
    ----------
    x, y, z : ~astropy.units.Quantity
        Geocentric cartesian coordinates.
    origin : tuple of ~astropy.units.Quantity
        Ggeocentric (spherical) or geodetic coordinates of the origin
        (`lat0`, `lon0`, `r0`) or (`lat0`, `lon0`, `h0`).
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`, optional
        Reference ellipsoid to which geodetic coordinates are referenced to.
        Default is None, meaning spherical coordinates instead of geodetic.

    Returns
    -------
    x, y, z : ~astropy.units.Quantity
        Local east-north-up cartesian coordinates.
    """
    rotation_matrix = _ecef_to_enu_rotation_matrix(origin[0], origin[1])
    if ell is None:
        x0, y0, z0 = spherical_to_cartesian(*origin)
    else:
        x0, y0, z0 = geodetic_to_cartesian(*origin, ell=ell)

    out_shape = x.shape

    xyz_shifted = _np.array([
        _np.asarray((x - x0).to('m').value).flatten(),
        _np.asarray((y - y0).to('m').value).flatten(),
        _np.asarray((z - z0).to('m').value).flatten()])

    out = _np.dot(rotation_matrix, xyz_shifted) * u.m

    return (out[0].reshape(out_shape),
            out[1].reshape(out_shape),
            out[2].reshape(out_shape))


@u.quantity_input
def enu_to_ecef(x: u.m, y: u.m, z: u.m, origin: tuple[u.deg, u.deg, u.m], ell=None):
    """Convert local cartesian to geocentric cartesian coordinates.

    Convert local east-north-up (ENU) local cartesian
    coordinate system with the origin in (`lat0`, `lon0`, `height0`)
    to the Earth Centered Earth Fixed (ECEF) cartesian coordinates.

    Parameters
    ----------
    x, y, z : ~astropy.units.Quantity
        Local east-north-uo cartesian coordinates.
    origin : tuple of ~astropy.units.Quantity
        Ggeocentric (spherical) or geodetic coordinates of the origin
        (`lat0`, `lon0`, `r0`) or (`lat0`, `lon0`, `h0`).
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.
        Default is None, meaning spherical coordinates instead of geodetic.

    Returns
    -------
    x, y, z : ~astropy.units.Quantity
        Geocentric cartesian coordinates, in metres.
    """
    rotation_matrix = _ecef_to_enu_rotation_matrix(origin[0], origin[1]).T

    out_shape = x.shape

    x, y, z = _np.dot(rotation_matrix, _np.array([
        _np.asarray(x.to('m').value).flatten(),
        _np.asarray(y.to('m').value).flatten(),
        _np.asarray(z.to('m').value).flatten()])) * u.m

    if ell is None:
        x0, y0, z0 = spherical_to_cartesian(*origin)
    else:
        x0, y0, z0 = geodetic_to_cartesian(*origin, ell=ell)

    return ((x + x0).reshape(out_shape),
            (y + y0).reshape(out_shape),
            (z + z0).reshape(out_shape))


@u.quantity_input
def geodetic_to_enu(lat: u.deg, lon: u.deg, height: u.m,
                    origin: tuple[u.deg, u.deg, u.m], ell):
    """Convert geodetic coordinates to local cartesian coordinates.

    Convert geodetic coordinates
    (`lat`,`lon`,`height`) to the local east-north-up (ENU) local cartesian
    coordinate system with the origin in (`lat0`, `lon0`, `height0`).

    Parameters
    ----------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    origin : tuple of ~astropy.units.Quantity
        Geodetic coordinates of the origin (`lat0`, `lon0`, `h0`).
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    x, y, z : ~astropy.units.Quantity
        Local east-north-up cartesian coordinates.
    """
    return ecef_to_enu(
        *geodetic_to_cartesian(lat, lon, height, ell=ell), origin=origin, ell=ell)


@u.quantity_input
def enu_to_geodetic(x: u.m, y: u.m, z: u.m, origin: tuple[u.deg, u.deg, u.m], ell):
    """Convert local cartesian coordinates to geodetic coordinates.

    Convert the local east-north-up (ENU) local cartesian
    coordinate system with the origin in (`lat0`, `lon0`, `height0`)
    to the geodetic coordinates.

    Parameters
    ----------
    x, y, z : ~astropy.units.Quantity
        Local east-north-uo cartesian coordinates.
    origin : tuple of ~astropy.units.Quantity
        Geodetic coordinates of the origin (`lat0`, `lon0`, `h0`).
    ell : instance of the `pygeoid.coordinates.ellipsoid.Ellipsoid`
        Reference ellipsoid to which geodetic coordinates are referenced to.

    Returns
    -------
    lat : ~astropy.units.Quantity
        Geodetic latitude.
    lon : ~astropy.units.Quantity
        Geodetic longitude.
    height : ~astropy.units.Quantity
        Geodetic height.
    """
    return cartesian_to_geodetic(
        *enu_to_ecef(x, y, z, origin=origin, ell=ell), ell=ell)

##############################################################################
# 2D coordinates
##############################################################################


@u.quantity_input
def polar_to_cartesian(theta: u.deg, radius: u.m):
    """Convert polar coordinates to 2D cartesian.

    Parameters
    ----------
    theta : ~astropy.units.Quantity
        Polar angle.
    radius : ~astropy.units.Quantity
        Radius.

    Returns
    -------
    x, y : ~astropy.units.Quantity
        Local east-north-uo cartesian coordinates.
    """
    return radius * _np.cos(theta), radius * _np.sin(theta)


@u.quantity_input
def cartesian_to_polar(x: u.m, y: u.m):
    """Convert 2D cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x, y : ~astropy.units.Quantity
        Cartesian coordinates.

    Returns
    -------
    theta : ~astropy.units.Quantity
        Polar angle.
    radius : ~astropy.units.Quantity
        Radius.
    """

    radius = _np.sqrt(x ** 2 + y ** 2)
    theta = _np.arctan2(y, x)

    return theta, radius
