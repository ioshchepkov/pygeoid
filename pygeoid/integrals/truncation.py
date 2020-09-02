"""Truncation coefficients for Stokes's integral.

"""

import numpy as np
import scipy.integrate as spi

from pyshtools.legendre import PLegendre

from pygeoid.integrals.stokes import StokesKernel

__all__ = ['molodensky_truncation_coefficients', 'paul_coefficients']


def _qn_hagiwara(t, n_max):
    """Molodensky's truncation coefficients by Hagiwara (1976).

    """
    p = PLegendre(n_max + 2, t)

    n = np.arange(n_max + 2)
    In = (p[n + 1] - p[n - 1]) / (2 * n + 1)

    n = np.arange(n_max + 1)
    J = ((n + 1) * In[n + 1] + n * In[n - 1]) / (2 * n + 1)

    K = np.empty(n_max + 1)
    K[0] = -0.5 * (1 - np.sqrt(2 / (1 - t)))
    K[1] = K[0] - (1 - np.sqrt(0.5 * (1 - t)))

    n = np.arange(2, n_max + 1)
    for i in n:
        K[i] = -In[i - 1] / np.sqrt(2 * (1 - t)) + 2 * K[i - 1] - K[i - 2]

    q = n * StokesKernel()._kernel_t(t) * (p[n - 1] - t * p[n])
    q -= (1 - t**2) * p[n] * StokesKernel()._derivative_t(t)
    q += 2 * K[n] + 2 * In[n] + 9 * J[n]
    q *= -1 / ((n - 1) * (n + 2))

    return q


def _qn_numerical(t, n_max, **kwargs):
    """Molodensky's truncation coefficients by numerical integration.

    """
    def f(x, n):
        return StokesKernel()._kernel_t(x) * PLegendre(n, x)[-1]

    q = [spi.quad(f, -1, t, args=(i,), **kwargs)[0]
         for i in range(2, n_max + 1)]

    return q


def molodensky_truncation_coefficients(
        spherical_distance: float,
        degree_n: int,
        method: str = 'hagiwara',
        **kwargs) -> np.ndarray:
    r"""Evaluate Molodensky's truncation coefficients Qn.

    Compute sequence of Molodensky's truncation coefficients for all degrees
    from 0 to degree_n (inclusive).

    Parameters
    ----------
    spherical_distance : float
        Spherical distance, in degrees.
    degree_n : int
        Maximum degree of the coefficients.
    method : {'hagiwara', 'numerical'}, optional
        Controls how coefficients are calculated.

        * 'hagiwara' calculates coefficients by Hagiwara (1976) recurrence relations.

        * 'numerical' calculates coefficients by numerical integration.

        Default is 'hagiwara'.

    **kwargs
        Keyword arguments for `scipy.intagrate.quad` if `method` is
        'numerical'.

    Returns
    -------
    array_like of floats
        Molodensky's truncation coefficient for all degrees from 0 to degree_n
        (inclusive).

    Notes
    -----
    The Molodensky's truncation coefficients :math:`Q_n` of degree :math:`n`
    are defined as [1]_:

        .. math::
            Q_n \left(\psi_0\right) = \int\limits_{\psi_0}^{\pi}
            S\left(\psi\right) P_n \left(\cos{\psi} \right) \sin{\psi} d\psi,

    where :math:`S\left(\psi\right)` -- Stokes function,
    :math:`P_n` -- Legendre polynomial, :math:`\psi` -- spherical distance.

    The function calculates this integral by Hagiwara's [2]_ method or by
    the numerical integration with `scipy.intagrate.quad`.

    References
    ----------
    .. [1] Molodensky MS, Yeremeyev VF, Yurkina MI (1962) Methods for study
        of the external gravitational field and figure of the Earth.
        Translated from Russian, Isreali Programme for Scientific Translations,
        Jerusalem
    .. [2] Hagiwara Y (1976) A new formula for evaluating the truncation
        error coefficient. Bulletin Géodésique 50:131–135.

    """
    if degree_n < 0 or not isinstance(degree_n, int):
        raise ValueError('degree_n must be non-negative integer.')

    if spherical_distance == 0:
        q = np.array([2 / (i - 1) if i > 1 else 0 for i in range(degree_n + 1)])
    elif spherical_distance == 180:
        q = np.zeros(degree_n + 1)
    elif 0 < spherical_distance < 180:
        q = np.empty(degree_n + 1)
        t = np.sin(np.radians(0.5 * spherical_distance))
        q[0] = -4 * t + 5 * t**2 + 6 * t**3 - 7 * t**4 +\
            (6 * t**2 - 6 * t**4) * np.log(t * (1 + t))
        if degree_n > 0:
            log1 = np.log(t * (1 + t))
            log2 = 2 * np.log(1 + t)
            q[1] = -2 * t + 4 * t**2 + 28 / 3 * t**3 - 14 * t**4 - 8 * t**5 +\
                32 / 3 * t**6 + (6 * t**2 - 12 * t**4 + 8 * t**6) * log1 - log2
        if degree_n > 1:
            t = np.cos(np.radians(spherical_distance))
            if method == 'hagiwara':
                q[2:] = _qn_hagiwara(t, degree_n)
            elif method == 'numerical':
                q[2:] = _qn_numerical(t, degree_n, **kwargs)
            else:
                raise ValueError('method must be `hagiwara` or `numerical`')
    else:
        raise ValueError('spherical_distance not in range: 0 <= psi <= 180 ')

    return q


def _rnk_tril_paul(r, t):
    p = PLegendre(max(r.shape) + 1, t)

    r[0][0] = t + 1
    r[1][1] = (t**3 + 1) / 3

    n, k = np.tril_indices_from(r, -1)

    # non-diagonal elements
    def w(x):
        return x * (x + 1) / (2 * x + 1)
    w1 = w(n) * p[k] * (p[n + 1] - p[n - 1])
    w2 = w(k) * p[n] * (p[k + 1] - p[k - 1])
    r[n, k] = (w1 - w2) / ((n - k) * (n + k + 1))

    # diagonal elements
    for n in np.arange(2, r.shape[0] - 1):
        w1 = (n + 1) * (2 * n - 1) / (n * (2 * n + 1))
        w2 = (n - 1) / n
        w3 = (2 * n - 1) / (2 * n + 1)
        r[n, n] = w1 * r[n + 1, n - 1] - w2 * r[n,
                                                n - 2] + w3 * r[n - 1, n - 1]
    return r


def _rnk_tril_num(r, t):
    for i, j in zip(*np.tril_indices_from(r)):
        if i != j:
            def f(x, ni, ki):
                return PLegendre(ni, x)[-1] * PLegendre(ki, x)[-1]
            args = (i, j)
        else:
            def f(x, ni):
                return PLegendre(ni, x)[-1]**2
            args = (i,)
        r[i, j] = spi.quad(f, -1, t, args=args, limit=1000)[0]

    return r


def paul_coefficients(
        spherical_distance: float,
        n: int,
        k: int = None,
        method: str = 'paul',
        **kwargs) -> np.ndarray:
    r"""Return Paul's coefficients.

    In the original article (1973) the Paul's coefficients are denoted as Rnk,
    but in many later articles they are denoted as enk.

    Parameters
    ----------
    spherical_distance : float or array_like of floats
        Spherical distance.
    n,k : int
        Degrees of the coefficients. `k` by default is None,
        i.e. it is equal to `n`.
    method : {'paul', 'numerical'}, optional
        Controls how coefficients are calculated.

        * 'paul' calculate coefficients by Paul (1973) recurrence relations.

        * 'numerical' calculate coefficients by numerical integration.

        Default is 'paul'.

    **kwargs
        Keyword arguments for `scipy.intagrate.quad` if `method` is
        'numerical'.

    Notes
    -----
    The Pauls's coefficients :math:`e_{nk}` of degrees :math:`n` and :math:`k`
    are defined as  [1]_:

        .. math::
            e_{nk} \left(\psi_0\right) =
            \int\limits_{\psi_0}^{\pi}
            P_n \left(\cos{\psi}\right) P_k \left(\cos{\psi}\right)
            \sin{\psi} d\psi,


    where :math:`P_n` and :math:`P_k` are Legendre polynomial of degrees
    :math:`n` and :math:`k` respectively, :math:`\psi` is the spherical
    distance.

    Note that in the original article [1]_ the Paul's coefficients are denoted as
    :math:`R_{n,k}`.

    References
    ----------
    .. [1] Paul MK (1973) A method of evaluating the truncation error
        coefficients for geoidal height. Bull Géodésique 110:413–425

    """

    psi = spherical_distance

    if 0 <= psi <= 180:
        t = np.cos(np.radians(psi))
    else:
        raise ValueError('psi not in range: 0 <= psi <= 180 ')

    if k is None:
        k = n
    if n < 0 or k < 0:
        raise ValueError('n and k must be >= 0')

    dmax = max(n, k)
    r = np.zeros((dmax + 2, dmax + 2))

    if method == 'paul':
        r = _rnk_tril_paul(r, t)
    elif method == 'numerical':
        r = _rnk_tril_num(r, t)

    # smart symmetry
    r = r + r.T - np.diag(r.diagonal())

    return r[:n + 1, :k + 1]
