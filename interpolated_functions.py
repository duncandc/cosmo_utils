"""
interpolated cosmology utility functions
"""

from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
import numpy as np
from default_cosmo import default_cosmo  # define a default cosology for utilities

__all__=['interpolated_distmod', 'interpolated_lookback_time']
__author__ = ['Duncan Campbell']


def interpolated_distmod(z, cosmo=None, dz=1e-5):
    """
    calculate the distance modulus for an array of redshifts using an
    interpolated function

    Paramaters
    ----------
    z : array_like
        array of redshifts

    cosmo : astropy.cosmnology object
        astropy cosmology object indicating cosmology to use

    dz : float
        redshift spacing when building interpolation function
    """

    z = np.atleast_1d(z)

    if np.min(z)<0.0:
        msg = ('all `z` must be greater or equal to 0.0.')
        assert ValueError(msg)

    if cosmo is None:
        cosmo = default_cosmo

    z_range = np.max(z) - np.min(z)
    n_sample = z_range/dz

    # require n_sample >= 2
    if np.min(n_sample)<2:
        n_sample=2

    # require minimum z for interpolation to be > 0.0
    if np.min(z)==0.0:
        min_z = dz
    else:
        min_z = np.min(z)

    z_sample = np.linspace(min_z, np.max(z), n_sample)

    d = cosmo.distmod(z_sample)

    f = interpolate.InterpolatedUnivariateSpline(z_sample, d, k=1, ext=3, check_finite=True)

    return np.where(z>0.0, f(z), 0.0)


def interpolated_lookback_time(z, cosmo='default', dz=1e-5):
    """
    lookback time

    Paramaters
    ----------
    z : array_like
        redshift

    cosmo : astropy.cosmology object

    Returns
    -------
    t : array_like
        lookback time to redshift z in h^-1 Gyr

    Notes
    -----
    This function builds an interpolation function instead of doing an integral for
    each z which makes this substantially faster than astropy.cosmology lookback_time()
    routine for large arrays.
    """

    z = np.atleast_1d(z)

    if np.min(z)<0.0:
        msg = ('all `z` must be greater or equal to 0.0.')
        assert ValueError(msg)

    if cosmo is None:
        cosmo = default_cosmo

    z_range = np.max(z) - np.min(z)
    n_sample = z_range/dz

    # require n_sample >= 2
    if np.min(n_sample)<2:
        n_sample=2

    # require minimum z for interpolation to be > 0.0
    if np.min(z)==0.0:
        min_z = dz
    else:
        min_z = np.min(z)

    z_sample = np.linspace(min_z, np.max(z), n_sample)

    t_sample = cosmo.lookback_time(z_sample).to('Gyr').value

    f = interp1d(np.log10(1.0+z_sample), np.log10(t_sample), fill_value='extrapolate')

    # set z=0 to t=0.0
    mask = (z>0.0)
    t = np.zeros(len(z))
    t[mask] = 10.0**f(np.log10(1+z[mask])) * cosmo.h

    return t

