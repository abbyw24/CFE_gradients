import numpy as np
import os
import sys

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import convert_3d_counts_to_cf
from colossus.cosmology import cosmology


def cf_model(r, cosmo_base=None, redshift=0.0, bias=1.0):

    if cosmo_base is None:
        cosmo_base = cosmology.setCosmology('planck15')

    cf = cosmo_base.correlationFunction

    return bias**2 * cf(r, z=redshift)


def cosmo_bases(rmin, rmax, projfn, cosmo_base=None, ncont=2000, 
              redshift=0.0, bias=1.0):
    """Returns basis function to use in Suave given a cosmological model."""

    if cosmo_base is None:
        # print("cosmo_base not provided, defaulting to Planck 2015 cosmology ('planck15')")
        cosmo_base = cosmology.setCosmology('planck15')

    rcont = np.linspace(rmin, rmax, ncont)
    bs = cf_model(rcont, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    nbases = 1
    bases = np.empty((ncont, nbases+1))
    bases[:,0] = rcont
    bases[:,1] = np.array(bs)

    np.savetxt(projfn, bases)
    return bases


def suave(x, y, z, xr, yr, zr, projfn, nthreads, r_edges, mumax, nmubins, boxsize, periodic, proj_type):
    """Compute the continuous 2pcf given an input (x,y,z) data set and basis functions."""

    # size of input data
    nd = len(x)
    nr = len(xr)

    # basis
    basis = np.loadtxt(projfn)
    r_fine = basis[:,0]
    ncomponents = basis.shape[1]-1

    # run the pair counts
    dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                                    boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                                    ncomponents=ncomponents, projfn=projfn)
    dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z,
                                    X2=xr, Y2=yr, Z2=zr,
                                    boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                                    ncomponents=ncomponents, projfn=projfn)
    rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins,
                                        xr, yr, zr, boxsize=boxsize,
                                        periodic=periodic, proj_type=proj_type,
                                        ncomponents=ncomponents, projfn=projfn)
    
    amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
    xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

    # results
    results = np.empty((ncont, 2))
    results[:,0] = r_fine
    results[:,1] = xi_proj

    return results