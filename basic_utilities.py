"""
cosmology utility functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const
import scipy.integrate as integrate

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


def hubble_distance(H0):
    """
    Calculate the Hubble distance
    
    parameters
    ----------
    H0: float
        Hubble constant in km/s/Mpc
    
    returns
    -------
    DH: float
        Hubble distance in Mpc
    """
    
    return const.c.to('km/s')/H0


def comoving_distance(z,cosmo=None):
    """
    Calculate the line of sight comoving distance
    
    parameters
    ----------
    z: float
        redshift
    
    cosmo: astropy.cosmology object, optional
        cosmology object specifying cosmology.  If None,  FlatLambdaCDM(H0=70,Om0=0.3)
    
    returns
    -------
    DC: float
        Comoving line of sight distance in Mpc
    """
    
    if cosmo==None:
        cosmo = default_cosmo
    
    f = lambda zz: 1.0/_Ez(zz, cosmo.Om0, cosmo.Ok0, cosmo.Ode0)
    DC = integrate.quadrature(f,0.0,z)[0]
    
    return hubble_distance(cosmo.H0.value)*DC


def transverse_comoving_distance(z,cosmo=None):
    """
    Calculate the transverse comoving distance
    
    parameters
    ----------
    z: float
        redshift
    
    cosmo: astropy.cosmology object, optional
        cosmology object specifying cosmology.  If None,  FlatLambdaCDM(H0=70,Om0=0.3)
    
    returns
    -------
    DM: float
        Comoving transverse distance in Mpc
    """
    
    if cosmo==None:
        cosmo = default_cosmo
    
    if cosmo.Ok0==0.0:
        return comoving_distance(z,cosmo)
    elif cosmo.Ok0>0:
        DC = comoving_distance(z,cosmo)
        DH = hubble_distance(cosmo.H0.value)
        return DH*1.0/np.sqrt(cosmo.Ok0)*np.sinh(np.sqrt(cosmo.Ok0)*DC/DH)
    elif cosmo.Ok0<0:
        DC = comoving_distance(z,cosmo)
        DH = hubble_distance(cosmo.H0.value)
        return DH*1.0/np.sqrt(np.fabs(cosmo.Ok0))*np.sin(np.sqrt(np.fabs(cosmo.Ok0))*DC/DH)
    else:
        raise ValueError("omega curavture value not specified.")


def angular_diameter_distance(z,cosmo=None):
    """
    Calculate the angular diameter distance
    
    parameters
    ----------
    z: float
        redshift
    
    cosmo: astropy.cosmology object, optional
        cosmology object specifying cosmology.  If None,  FlatLambdaCDM(H0=70,Om0=0.3)
    
    returns
    -------
    DA: float
        Angular diameter distance in Mpc
    """

    if cosmo==None:
        cosmo = default_cosmo

    return transverse_comoving_distance(z,cosmo)/(1.0+z)


def luminosity_distance(z,cosmo=None):
    """
    Calculate the luminosity distance.
    
    parameters
    ----------
    z: float
        redshift
    
    cosmo: astropy.cosmology object, optional
        cosmology object specifying cosmology.  If None,  FlatLambdaCDM(H0=70,Om0=0.3)
    
    returns
    -------
    DL: float
        Luminosity distance in Mpc
    """

    if cosmo==None:
        cosmo = default_cosmo

    return transverse_comoving_distance(z,cosmo)*(1.0+z)


def comoving_volume(z,dw,cosmo=None):
    """
    Calculate comoving volume
    
    parameters
    ----------
    z: float
        redshift
    
    dw: float
        solid angle
    
    cosmo: astropy.cosmology object, optional
        cosmology object specifying cosmology.  If None,  FlatLambdaCDM(H0=70,Om0=0.3)
    
    returns
    -------
    VC: float
        comoving volume in Mpc^3
    """

    if cosmo==None:
        cosmo = default_cosmo

    DH = hubble_distance(cosmo.H0.value) 
    f = lambda zz: DH*((1.0+zz)**2.0*angular_diameter_distance(zz,cosmo)**2.0)/(_Ez(zz, cosmo.Om0, cosmo.Ok0, cosmo.Ode0))

    VC = integrate.quadrature(f,0.0,z,vec_func=False)[0]*dw
    
    return VC


def _Ez(z, omega_m, omega_k, omega_l):
    """
    internal function used for distance calculations
    """
    return np.sqrt(omega_m*(1.0+z)**3.0+omega_k*(1.0+z)**2.0+omega_l)



