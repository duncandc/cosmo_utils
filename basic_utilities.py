"""
cosmology utility functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const

__all__=('mean_density',)
__author__=('Duncan Campbell')

# define a default cosology for utilities
from astropy.cosmology import FlatLambdaCDM
default_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)


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

