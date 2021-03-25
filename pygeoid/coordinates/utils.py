
import numpy as np
import astropy.units as u


def spherical_distance(lat1: u.deg, lon1: u.deg, lat2: u.deg, lon2: u.deg):
    """Return distance on the sphere.

    Returns distances between initial
    points (specified by `lat1`, `lon1`) and terminus points (specified by
    `lat1`, `lon2`) as a distance matrix.

    Parameters
    ----------
    lat1 : ~astropy.units.Quantity
        Geocentric latitude of the initial points.
    lon1 : ~astropy.units.Quantity
        Longitude of the initial points.
    lat2 : ~astropy.units.Quantity
        Geocentric latitude of the terminus points.
    lon2 : ~astropy.units.Quantity
        Longitude of the terminus points.

    Returns
    -------
    distance : ~astropy.units.Quantity
        Distance matrix.
    """

    lat1m, lat2m = np.meshgrid(lat1, lat2)
    lon1m, lon2m = np.meshgrid(lon1, lon2)

    lat_dif = lat1m - lat2m
    lon_dif = lon1m - lon2m

    slatsq = np.sin(0.5 * lat_dif)**2
    slonsq = np.sin(0.5 * lon_dif)**2
    spsi2 = np.sqrt(slatsq + slonsq * np.cos(lat1m) * np.cos(lat2m))

    psi = 2 * np.arcsin(spsi2)

    return psi
