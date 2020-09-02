
import numpy as np


def spherical_distance(lat1, lon1, lat2, lon2, degrees=True):
    """Return distance on the sphere.

    Returns distances between initial
    points (specified by `lat1`, `lon1`) and terminus points (specified by
    `lat1`, `lon2`) as a distance matrix.

    Parameters
    ----------
    lat1 : float or array_like of floats
        Geocentric latitude of the initial points.
    lon1 : float or array_like of floats
        Longitude of the initial points.
    lat2 : float or array_like of floats
        Geocentric latitude of the terminus points.
    lon2 : float or array_like of floats
        Longitude of the terminus points.
    degrees : bool, optional
        If True, the input `lat1`, `lon1`, `lat2`, `lon2`
        are given in degrees, otherwise radians.
        Default is True.

    Returns
    -------
    distance : array_like of floats
        Distance matrix, in degrees or radians.
    """

    lat1m, lat2m = np.meshgrid(lat1, lat2)
    lon1m, lon2m = np.meshgrid(lon1, lon2)

    lat_dif = lat1m - lat2m
    lon_dif = lon1m - lon2m

    if degrees:
        lat1m = np.radians(lat1m)
        lat2m = np.radians(lat2m)

        lat_dif = np.radians(lat_dif)
        lon_dif = np.radians(lon_dif)

    slatsq = np.sin(0.5 * lat_dif)**2
    slonsq = np.sin(0.5 * lon_dif)**2
    spsi2 = np.sqrt(slatsq + slonsq * np.cos(lat1m) * np.cos(lat2m))

    psi = 2 * np.arcsin(spsi2)

    if degrees:
        psi = np.degrees(psi)

    return psi
