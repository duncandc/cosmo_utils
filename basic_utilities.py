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


def lookback_time(z, cosmo='default'):
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
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    
    #if z<0, set t=0.0
    mask = (z>0.0)
    t = np.zeros(len(z))
    
    #build interpolation function for t_look(z)
    max_z = np.max(z)
    z_sample = np.logspace(0,np.log10(max_z+1),1000) - 1.0
    t_sample = cosmo.lookback_time(z_sample).to('Gyr').value
    f = interp1d(np.log10(1+z_sample), np.log10(t_sample), fill_value='extrapolate')
    
    t[mask] = 10.0**f(np.log10(1+z[mask])) * cosmo.h
    
    return t



