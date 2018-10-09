"""
default cosmology object for use in cosmo_utils module
"""

from astropy.cosmology import FlatLambdaCDM

default_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)
