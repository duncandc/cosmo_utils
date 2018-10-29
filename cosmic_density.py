"""
density calculations
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const
import scipy.integrate as integrate
from default_cosmo import default_cosmo  # define a default cosology for utilities


__all__=('mean_density', 'critical_density',)
__author__=('Duncan Campbell')


def mean_density(z, cosmo=None):
    """
    Return the mean density of the Universe.

    Paramaters
    ----------
    z : array_like
        arry of redshifts

    cosmo : astropy.cosmology object
        cosmology object

    Returns
    -------
    rho_b : numpy.array
         mean density of the universe at redshift z in units Msol/Mpc^3
    """

    if cosmo is None:
        cosmo = default_cosmo

    z = np.atleast_1d(z)
    a = 1.0/(1.0+z)  # scale factor

    rho = (3.0/(8.0*np.pi*const.G))*(cosmo.H(z)**2)*(cosmo.Om(z)*a**(-3.0))
    rho = rho.to(u.M_sun / u.parsec**3.0)*((10**6)**3.0)

    return rho.value


def critical_density(z, cosmo='default'):
    """
    critical density of the universe

    Paramaters
    ----------
    z : array_like
        redshift

    cosmo : astropy.cosmology object
        cosmology object

    Returns
    -------
    rho_c : numpy.array
        critical density of the universe at redshift z in g/cm^3
    """

    if cosmo=='default':
        cosmo = default_cosmo

    rho = (3.0*cosmo.H(z)**2)/(8.0*np.pi*const.G)
    rho = rho.to(u.g / u.cm**3)

    return rho
