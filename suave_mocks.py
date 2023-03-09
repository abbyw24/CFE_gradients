import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import datetime

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import convert_3d_counts_to_cf
from colossus.cosmology import cosmology

from funcs.suave import cf_model, cosmo_bases
from funcs.center_data import center_data
import funcs.random_cat

import globals
globals.initialize_vals()  # brings in all the default parameters

# function to run suave with any basis
def suave(x, y, z, L, n, projfn,
            proj_type = 'generalr',
            load_rand = True,
            data_dir = globals.data_dir,
            randmult = globals.randmult,
            periodic = globals.periodic,
            rmin = globals.rmin,
            rmax = globals.rmax,
            nbins = globals.nbins,
            ncont = globals.ncont,
            nthreads = globals.nthreads,
            nmubins = 1,
            mumax = 1.0
            ):
    """Use Suave to compute the continuous 2pcf given an input data set and basis file."""

    # data
    data = np.array([x, y, z])
    center_data(data, 0, L)
    nd = len(x)

    # other parameters for suave
    r_edges = np.linspace(rmin, rmax, nbins+1) 
    r_fine = np.linspace(rmin, rmax, ncont)

    # random set: either load a pre-computed set, or generate one here
    if load_rand:
        try:
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{int(L)}_n{n}_{randmult}x.dat')
        except OSError: # generate the random catalog if it doesn't already exist
            random_cat.main(L, n, data_dir, randmult)
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{int(L)}_n{n}_{randmult}x.dat')
        finally:
            rand_set = np.loadtxt(random_fn)
        # rand_set.shape == (nr, 3)
    else:
        nr = randmult * nd
        rand_set = np.random.uniform(0, L, (int(nr),3))
    center_data(rand_set, 0, L)
    xr, yr, zr = rand_set.T
    nr = len(xr)

    # basis
    basis = np.loadtxt(projfn)
    ncomponents = basis.shape[1]-1

    # run the pair counts
    dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                                    boxsize=L, periodic=periodic, proj_type=proj_type,
                                    ncomponents=ncomponents, projfn=projfn)
    dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z,
                                    X2=xr, Y2=yr, Z2=zr,
                                    boxsize=L, periodic=periodic, proj_type=proj_type,
                                    ncomponents=ncomponents, projfn=projfn)
    rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins,
                                            xr, yr, zr, boxsize=L,
                                            periodic=periodic, proj_type=proj_type,
                                            ncomponents=ncomponents, projfn=projfn)
    
    amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
    xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

    # results
    results = np.empty((ncont, 2))
    results[:,0] = r_fine
    results[:,1] = xi_proj

    return results



# function to estimate gradient using suave
def suave_grad(x, y, z, L, n, projfn,
                load_rand = True,
                data_dir = globals.data_dir,
                randmult = globals.randmult,
                proj_type = 'gradient',
                weight_type = 'pair_product_gradient',
                periodic = globals.periodic,
                rmin = globals.rmin,
                rmax = globals.rmax,
                nbins = globals.nbins,
                ncont = globals.ncont,
                nthreads = globals.nthreads,
                nmubins = 1,
                mumax = 1.0,
                compute_standard = False):
    """Use Suave to estimate the gradient in clustering amplitude given an input data set and basis file."""

    # create suave dictionary
    suave_dict = {}

    # parameters
    nd = len(x)
    r_edges = np.linspace(rmin, rmax, nbins+1) 
    r_fine = np.linspace(rmin, rmax, ncont)

    # random set: either load a pre-computed set, or generate one here
    if load_rand:
        try:
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{int(L)}_n{n}_{randmult}x.dat')
        except OSError: # generate the random catalog if it doesn't already exist
            random_cat.main(L, n, data_dir, randmult)
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{int(L)}_n{n}_{randmult}x.dat')
        finally:
            rand_set = np.loadtxt(random_fn)
        # rand_set.shape == (nr, 3)
    else:
        nr = randmult * nd
        rand_set = np.random.uniform(0, L, (int(nr),3))
    center_data(rand_set, 0, L)
    xr, yr, zr = rand_set.T
    nr = len(xr)

    # basis
    basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)
    ncomponents = 4*(basis.shape[1]-1)
    
    # weights
    loc_pivot = [L/2., L/2., L/2.]
    weights = np.array([np.ones(len(x)), x-loc_pivot[0], y-loc_pivot[1], z-loc_pivot[2]])
    weights_r = np.array([np.ones(len(xr)), xr-loc_pivot[0], yr-loc_pivot[1], zr-loc_pivot[2]])

    # run the pair counts
    dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
                            proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                            periodic=periodic, weight_type=weight_type)

    dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
                            X2=xr, Y2=yr, Z2=zr, weights2=weights_r, 
                            proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                            periodic=periodic, weight_type=weight_type)

    rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, xr, yr, zr, weights1=weights_r, 
                                    proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                                    periodic=periodic, weight_type=weight_type)

    amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
    xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

    # extract the standard binned values, if specified
    if compute_standard:
        dd = np.array([x['npairs'] for x in dd_res], dtype=float)
        dr = np.array([x['npairs'] for x in dr_res], dtype=float)
        rr = np.array([x['npairs'] for x in rr_res], dtype=float)
        xi_standard = convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
        r_avg = 0.5*(r_edges[:-1] + r_edges[1:])
        # add these results to the results dictionary
        suave_dict['r_avg'] = r_avg
        suave_dict['xi_standard'] = xi_standard

    # save other plot parameters
    suave_dict['amps'] = amps
    suave_dict['r_fine'] = r_fine
    suave_dict['proj_type'] = proj_type
    suave_dict['xi_proj'] = xi_proj
    suave_dict['projfn'] = projfn
    suave_dict['weight_type'] = weight_type
    suave_dict['periodic'] = periodic
    suave_dict['randmult'] = randmult

    return suave_dict