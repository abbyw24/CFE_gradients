"""
GOAL: convert a set of input (RA, Dec, z) data to (x,y,z) data

1. redshift (z) --> comoving distance (r)
2. (RA, Dec) --> (theta, phi)
3. (r, theta, phi) --> (x, y, z)

Steps 2-3 are simple coordinate transformations, while step 1 requires a cosmological model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import astropy


def z_to_r_comov(z, cosmo=astropy.cosmology.Planck15):
    r = (cosmo.comoving_distance(z)*cosmo.h).value # convert to Mpc/h
    return r

def radec_to_thetaphi(ra, dec):
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180
    return theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def convert_radecz_to_xyz(ra, dec, z):
    assert len(ra)==len(dec)==len(z), "Inputs must have the same length!"
    ndata = len(ra)
    xyz_arr = np.empty((ndata, 3))
    for i in range(ndata):
        D_comov = z_to_r_comov(z[i])
        theta, phi = radec_to_thetaphi(ra[i], dec[i])
        xyz_arr[i] = spherical_to_cartesian(D_comov, theta, phi)
    return xyz_arr