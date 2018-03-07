"""Calculate atmospheric correction for the gravity anomalies."""

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from pygeoid.constants import G


def ussa76_density(alt_arr=0.0):
    """Return atmospheric density from USSA76 model.

    Refer to the following document (2 doc codes) for details of this model:
        NOAA-S/T 76-1562
        NASA-TM-X-74335

    All assumptions are the same as those in the source documents.

    Derived from: https://github.com/mattljc/atmosphere-py
    """
    # Constants
    g0 = 9.80665
    Rstar = 8.31432e3

    # Model Parameters
    altitude_max = 84852
    base_alt = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0]) * 1000
    base_lapse = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0]) / 1000
    base_temp = np.array(
        [288.15, 216.65, 216.650, 228.650, 270.650, 270.650, 214.650])
    base_press = np.array([1.01325e3, 2.2632e2, 5.4748e1, 8.6801, 1.1090,
                           6.6938e-1, 3.9564e-2]) * 100
    M0 = 28.9644

    # Initialize Outputs
    alt_arr = np.atleast_1d(alt_arr)
    dens_arr = np.zeros_like(alt_arr)

    for idx in range(alt_arr.size):
        alt = alt_arr[idx]
        if alt > altitude_max:
            msg = 'Altitude exceeds the model: h > hmax = {} m'.format(
                altitude_max)
            raise ValueError(msg)

        # Figure out base height
        if alt <= 0.0:
            base_idx = 0
        elif alt > base_alt[-1]:
            base_idx = len(base_alt) - 1
        else:
            base_idx = np.searchsorted(base_alt, alt, side='left') - 1

        alt_base = base_alt[base_idx]
        temp_base = base_temp[base_idx]
        lapse_base = base_lapse[base_idx]
        press_base = base_press[base_idx]

        temp = temp_base + lapse_base * (alt_arr[idx] - alt_base)
        if lapse_base == 0.0:
            press = press_base * \
                np.exp(-g0 * M0 * (alt_arr[idx] -
                                   alt_base) / Rstar / temp_base)
        else:
            press = press_base * \
                (temp_base / temp) ** (g0 * M0 / Rstar / lapse_base)

        dens_arr[idx] = press * M0 / Rstar / temp

    return dens_arr


def iag_atm_corr_sph(density_function, height, height_max, samples=1e4):
    """Return atmospheric correction to the gravity anomalies by IAG approach.

    This function numerically integrates samples from density function by
    trapezoidal rule. The spherical layering of the atmosphere is considered.

    IAG approach:

    g_atm = G*M(r) / r**2

                inf
                /
    M(r) = 4*pi*| rho(r) * r**2 dr
                /
                r

    Parameters
    ----------
    density_function : callable
        The `density_funtion` is called for all height samples to calculate
        density of the atmosphere.
    height : float or array_like of floats
        Height above sea level, in metres.
    height_max : float or array_like of floats
        Maximum height of the atmosphere layer above sea level, in metres.
    samples : float
        Number of samples for integration. Default is 1e4.
    """
    Rearth = 6378e3
    r = (Rearth + height)**2
    hinf = np.linspace(height, height_max, samples)
    density = density_function(hinf) * r
    M = 4 * np.pi * trapz(density, hinf)
    gc = (G.magnitude * M / r) * 1e5
    return gc


def grs80_atm_corr_interp(height, kind='linear'):
    """Return GRS 80 atmospheric correction, in mGal.

    Interpolated from the table data [1]_.

    Note: If height < 0 m or height > 40000 m, then correction is extrapolated

    Parameters
    ----------
    height : float or array_like of floats
        Height above sea level, in metres.
    kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order) or as an
        integer specifying the order of the spline interpolator to use.
        Default is 'linear'.

    Returns
    -------
    float or array_like of floats
        Atmospheric correction, in mGal.

    References
    ----------
    .. [1] Moritz, H. (1980). Geodetic reference system 1980.
    Bulletin Géodésique, 54(3), 395-405
    """
    fname = os.path.join(os.path.dirname(__file__),
                         'data/IAG_atmosphere_correction_table.txt')
    table_heights, corr = np.loadtxt(fname, unpack=True, delimiter=',',
                                     skiprows=4, dtype=float)
    interp = interp1d(table_heights * 1000, corr, kind=kind,
                      fill_value='extrapolate', assume_sorted=True)
    return interp(height)


def wenzel_atm_corr(height):
    """Return atmospheric correction by Wenzel, in mGal.

    Parameters
    ----------
    height : float or array_like of floats
        Height above sea level, in metres.

    References
    ----------
    .. [1] Wenzel, H., 1985, Hochauflosende Kugelfunktionsmodelle fur des
    Gravitationspotential der Erde [1]: Wissenschaftliche arbeiten der
    Fachrichtung Vermessungswesen der Universitat Hannover, 137
    """
    return 0.874 - 9.9e-5 * height + 3.56e-9 * height**2


def pz90_atm_corr(height):
    """Return PZ-90 atmospheric correction, in mGal.

    Parameters
    ----------
    height : float or array_like of floats
        Height above sea level, in metres.
    """
    return 0.87 * np.exp(-0.116 * (height / 1e3)**(1.047))
