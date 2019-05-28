"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const
import scipy.integrate as integrate
from .default_cosmo import default_cosmo  # define a default cosology for utilities
from .distance_functions import hubble_distance, angular_diameter_distance, _Ez

__all__=('comoving_volume',)
__author__=('Duncan Campbell')


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
