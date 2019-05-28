"""
cosmology utility functions relating to dark matter haloes
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const
from .default_cosmo import default_cosmo  # define a default cosology for utilities


__all__=('delta_vir', 'virial_radius', 'virial_halo_mass',
         'halo_mass_conversion', 'dynamical_time')
__author__=('Duncan Campbell')


def delta_vir(z, cosmo='default', wrt='background'):
    """
    The average over-density of a collapsed dark matter halo. 
    fitting function from Bryan & Norman (1998)
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    delta_vir : numpy.array
        average density with respect to the mean density of the Universe
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    
    x = cosmo.Om(z)-1.0
    
    if wrt=='critical':
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)
    elif wrt=='background':
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)/cosmo.Om(z)


def virial_radius(m, z=0.0, cosmo='default'):
    """
    The virial radius of a collapsed dark matter halo
    
    Paramaters
    ----------
    m : array_like
        halo mass
    
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    r_vir : numpy.array
        virial radius in h^{-1}Mpc
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    m = (np.atleast_1d(m)*u.Msun)/cosmo.h
    
    dvir = delta_vir(z, cosmo, wrt='background')
    
    r =  (m / ((4.0/3.0)*np.pi*dvir*mean_density(z, cosmo)))**(1.0/3.0)
    r = r.to(u.Mpc)*cosmo.h
    return r.value


def virial_halo_mass(m_h, c, delta_h=200, z=0.0, cosmo='default',
                         wrt='background'):
    """
    Convert halo mass to virial halo mass
    fitting function from Hu \& Kravtsov (2003).
    
    Parameters
    ----------
    m_h : array_like
        halo mass
    
    c : array_like
        concentration
    
    delta_h : float
        density contrast
    
    delta_vir : float, optional
        virial over-density wrt the mean density.  If given, cosmology is ignored 
        when calculating delta_vir.
        
    cosmo : astropy.cosmology object, optional
        cosmology used to calculated the virial over-density
    
    wrt : string
        halo over-density wrt respect to the 
        'background' density or 'critical' density
    
    Returns
    -------
    m_vir : array_like
        virial halo mass in h^{-1}M_{\odot}
    """
    
    m_h = np.atleast_1d(m_h)
    c = np.atleast_1d(c)
    z = np.atleast_1d(z)
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    if wrt=='background':
        pass
    elif wrt=='critical':
        delta_h = delta_h/cosmo.Om(z)
    else:
        msg = 'mean density wrt paramater not recognized.'
        raise ValueError(msg)
    
    def f_x(x):
        """
        eq. C3
        """
        return x**3*(np.log(1.0+1.0/x)-1.0/(1.0+x))
    
    def x_f(f):
        """
        fitting function to inverse of f_x, eq. C11
        """
        
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13*10**(-3)
        a4 = -3.52*10**(-5)
        p = a2 +a3*np.log(f) + a4*np.log(f)**2
        
        return (a1*f**(2.0*p) + (3/4)**2)**(-0.5) + 2.0*f
    
    f_h = delta_h/delta_vir(z, cosmo)
    
    r_ratio = x_f(f_h*f_x(1.0/c))
    
    f = (f_h)*r_ratio**(-3)*(1.0/c)**3.0
    
    return m_h/f


def halo_mass_conversion(m_h, c, delta_h=200, z=0.0, cosmo='default',
                         wrt_h='background', delta_new=360.0, wrt_new='background'):
    """
    Converrt between halo mass definitions
    fitting function from Hu \& Kravtsov (2003).
    
    Parameters
    ----------
    m_h : array_like
        halo mass
    
    c : array_like
        concentration
    
    delta_h : float
        over-density
    
    cosmo : astropy.cosmology object, optional
        cosmology used to calculated the virial over-density
    
    wrt_h : string
        halo over-density wrt respect to the 
        'background' density or 'critical' density
    
    delta_new : float
        convert to over-density
        
    wrt_new : string
        covert to halo over-density wrt respect to the 
        'background' density or 'critical' density
    
    Returns
    -------
    m_vir : array_like
        virial halo mass in h^{-1}M_{\odot}
    """
    
    m_h = np.atleast_1d(m_h)
    c = np.atleast_1d(c)
    z = np.atleast_1d(z)
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    if wrt_h=='background':
        pass
    elif wrt_h=='critical':
        delta_h = delta_h/cosmo.Om(z)
    else:
        msg = 'mean density wrt paramater not recognized.'
        raise ValueError(msg)
        
    if wrt_new=='background':
        pass
    elif wrt_new=='critical':
        delta_new = delta_new/cosmo.Om(z)
    else:
        msg = 'mean density wrt paramater not recognized.'
        raise ValueError(msg)
    
    def f_x(x):
        """
        eq. C3
        """
        return x**3*(np.log(1.0+1.0/x)-1.0/(1.0+x))
    
    def x_f(f):
        """
        fitting function to inverse of f_x, eq. C11
        """
        
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13*10**(-3)
        a4 = -3.52*10**(-5)
        p = a2 +a3*np.log(f) + a4*np.log(f)**2
        
        return (a1*f**(2.0*p) + (3/4)**2)**(-0.5) + 2.0*f
    
    f_h = delta_h/delta_vir(z, cosmo)
    
    r_ratio = x_f(f_h*f_x(1.0/c))
    
    f_1 = (f_h)*r_ratio**(-3)*(1.0/c)**3.0
    
    f_h = delta_new/delta_vir(z, cosmo)
    
    r_ratio = x_f(f_h*f_x(1.0/c))
    
    f_2 = (f_h)*r_ratio**(-3)*(1.0/c)**3.0
    
    f = f_1/f_2
    
    return m_h/f


def dynamical_time(z, cosmo='default'):
    """
    dynamical time
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    t : array_like
        dynamical time at redshift z h^-1 Gyr
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    
    t_dyn = 1.628*((delta_vir(z, cosmo)/178.0)**(-0.5)) * ((cosmo.H(z)/cosmo.H0)**(-1.0))
    
    return t_dyn.value

