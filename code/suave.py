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

from center_mock import center_mock
import random_cat
import generate_mock_list
import globals

globals.initialize_vals()  # brings in all the default parameters



def cf_model(r, cosmo_base=None, redshift=0.0, bias=1.0):

    if cosmo_base is None:
        cosmo_base = cosmology.setCosmology('planck15')

    cf = cosmo_base.correlationFunction

    return bias**2 * cf(r, z=redshift)



# define cosmo_bases function
def cosmo_bases(rmin, rmax, projfn, cosmo_base=None, ncont=2000, 
              redshift=0.0, bias=1.0):

    if cosmo_base is None:
        print("cosmo_base not provided, defaulting to Planck 2015 cosmology ('planck15')")
        cosmo_base = cosmology.setCosmology('planck15')

    rcont = np.linspace(rmin, rmax, ncont)
    bs = cf_model(rcont, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    nbases = 1
    bases = np.empty((ncont, nbases+1))
    bases[:,0] = rcont
    bases[:,1] = np.array(bs)

    np.savetxt(projfn, bases)
    return bases



# function to run suave with any basis
def suave(x, y, z, L, n, projfn,
            proj_type = 'generalr',
            load_rand = True,
            randmult = globals.randmult,
            periodic = globals.periodic,
            rmin = globals.rmin,
            rmax = globals.rmax,
            ncont = globals.ncont,
            nthreads = globals.nthreads,
            nmubins = 1,
            mumax = 1.0
            ):
    """Use Suave to compute the continuous 2pcf given an input data set and basis file"""

    # data
    data = np.array([x, y, z])
    center_mock(data, 0, L)
    nd = len(x)

    # random set: either load a pre-computed set, or generate one here
    if load_rand:
        try:
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{L}_n{n}_{randmult}x.dat')
        except OSError: # generate the random catalog if it doesn't already exist
            random_cat.main(L, n, data_dir, randmult)
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{L}_n{n}_{randmult}x.dat')
        finally:
            rand_set = np.loadtxt(random_fn)
        # rand_set.shape == (nr, 3)
    else:
        nr = randmult * nd
        rand_set = np.random.uniform(0, L, (int(nr),3))
    center_mock(rand_set, 0, L)
    xr, yr, zr = xs_rand.T

    # other parameters for suave
    r_edges = np.linspace(rmin, rmax, nbins+1) 
    r_fine = np.linspace(rmin, rmax, ncont)

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
def suave_grad(x, y, z, L, projfn,
                load_rand = True,
                randmult = globals.randmult,
                proj_type = 'gradient',
                weight_type = 'pair_product_gradient',
                periodic = globals.periodic,
                rmin = globals.rmin,
                rmax = globals.rmax,
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
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{L}_n{n}_{randmult}x.dat')
        except OSError: # generate the random catalog if it doesn't already exist
            random_cat.main(L, n, data_dir, randmult)
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{L}_n{n}_{randmult}x.dat')
        finally:
            rand_set = np.loadtxt(random_fn)
        # rand_set.shape == (nr, 3)
    else:
        nr = randmult * nd
        rand_set = np.random.uniform(0, L, (int(nr),3))
    center_mock(rand_set, 0, L)
    xr, yr, zr = xs_rand.T

    # basis
    basis = np.loadtxt(projfn)
    ncomponents = basis.shape[1]-1
    
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

    # recovered gradient
    w_cont = amps[1:]/amps[0]
    suave_dict['grad_recovered'] = w_cont
    # w_cont_norm = np.linalg.norm(w_cont)
    # w_cont_hat = w_cont/w_cont_norm

    # save other plot parameters
    suave_dict['amps'] = amps
    suave_dict['r_fine'] = r_fine
    suave_dict['proj_type'] = proj_type
    suave_dict['projfn'] = projfn
    suave_dict['weight_type'] = weight_type

    return suave_dict
        
    

# OLD SUAVE GRADIENT FUNCTION
# def suave_grad(nmocks = globals.nmocks,
#                 L = globals.boxsize,
#                 n = globals.lognormal_density,
#                 As = globals.As,
#                 rlzs = globals.rlzs,
#                 data_dir = globals.data_dir,
#                 grad_dim = globals.grad_dim,
#                 m = globals.m,
#                 b = globals.b,
#                 rmin = globals.rmin,
#                 rmax = globals.rmax,
#                 ncont = globals.ncont,
#                 bao_fixed=True,         # ** fixed vs. iterative BAO basis
#                 prints=True,
#                 plots=False):

#     s = time.time()

#     # generate mock list
#     mock_set = generate_mock_list.mock_set(nmocks, L, n, As=As, data_dir=data_dir, rlzs=rlzs)
#     # add the desired gradient to this initial mock set
#     mock_set.add_gradient(grad_dim, m, b)
#     cat_tag = mock_set.cat_tag

#     # basis type: fixed vs iterative
#     basis_type = 'bao_fixed' if bao_fixed else 'bao_iterative'

#     # create path to suave results directory if it doesn't already exist
#     suave_dir = f'suave_data/{cat_tag}/{basis_type}'
#     if not os.path.exists(suave_dir):
#         os.makedirs(suave_dir)

#     # parameters for suave (not already imported from globals)
#     r_edges = np.linspace(rmin, rmax, nbins+1) 
#     r_fine = np.linspace(rmin, rmax, ncont)

#     proj_type = 'gradient'
#     weight_type = 'pair_product_gradient'

#     # basis if bao_fixed
#     if bao_fixed:
#         projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')
#         basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)
#         ncomponents = 4*(basis.shape[1]-1)

#     for i, rlz in enumerate(mock_set.rlzs):

#         mock_fn = mock_fn_list[i]

#         # load bases
#         if not bao_fixed:
#             projfn = os.path.join(data_dir, f'bases/bao_iterative/results/results_gradient_{cat_tag}/final_bases/basis_gradient_{mock_fn}_trrnum_{randmult}x.dat')
#             basis = np.loadtxt(projfn)
#             ncomponents = 4*(basis.shape[1]-1)


#         # load in mock and patch info
#         mock_dict = np.load(os.path.join(mock_set.grad_dir, f'mock_data/{cat_tag}/{mock_fn}.npy'), allow_pickle=True).item()
#         mock_data = mock_dict['grad_set']
#         L = mock_dict['boxsize']
#         grad_expected = mock_dict['grad_expected']

#         # center data points between 0 and L
#         center_mock(mock_data, 0, L)

#         # create suave dictionary
#         suave_dict = {}

#         nd = len(mock_data)
#         x, y, z = mock_data.T

#         # random set
#         nr = randmult*nd
#         xr = np.random.rand(nr)*float(L)
#         yr = np.random.rand(nr)*float(L)
#         zr = np.random.rand(nr)*float(L)

#         # weights
#         loc_pivot = [L/2., L/2., L/2.]
#         weights = np.array([np.ones(len(x)), x-loc_pivot[0], y-loc_pivot[1], z-loc_pivot[2]])
#         weights_r = np.array([np.ones(len(xr)), xr-loc_pivot[0], yr-loc_pivot[1], zr-loc_pivot[2]])

#         # run the pair counts
#         dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
#                                 proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
#                                 periodic=periodic, weight_type=weight_type)

#         dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
#                                 X2=xr, Y2=yr, Z2=zr, weights2=weights_r, 
#                                 proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
#                                 periodic=periodic, weight_type=weight_type)

#         rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, xr, yr, zr, weights1=weights_r, 
#                                         proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
#                                         periodic=periodic, weight_type=weight_type)

#         amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
#         xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

#         # extract the standard binned values
#         dd = np.array([x['npairs'] for x in dd_res], dtype=float)
#         dr = np.array([x['npairs'] for x in dr_res], dtype=float)
#         rr = np.array([x['npairs'] for x in rr_res], dtype=float)
#         xi_standard = convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
#         r_avg = 0.5*(r_edges[:-1] + r_edges[1:])

#         # recovered gradient
#         w_cont = amps[1:]/amps[0]
#         w_cont_norm = np.linalg.norm(w_cont)
#         w_cont_hat = w_cont/w_cont_norm
#         suave_dict['grad_recovered'] = w_cont

#         # save other plot parameters
#         suave_dict['r_avg'] = r_avg
#         suave_dict['amps'] = amps
#         suave_dict['r_fine'] = r_fine
#         suave_dict['proj_type'] = proj_type
#         suave_dict['projfn'] = projfn
#         suave_dict['weight_type'] = weight_type

#         # save suave dictionary
#         save_fn = os.path.join(grad_dir, f'{suave_dir}/{mock_fn}')
#         np.save(save_fn, suave_dict, allow_pickle=True)
        
#         if plots == True:
#             # define plot parameters
#             v_min = -L/2.
#             v_max = L/2.
#             nvs = 50
#             vs = np.linspace(v_min, v_max, nvs)
#             xi_locs = []

#             fig, ax = plt.subplots()
#             vs_norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
#             cmap = matplotlib.cm.get_cmap('cool')

#             # compute and plot xi at nvs evenly-spaced positions across the box
#             for i, v in enumerate(vs):
#                 loc = loc_pivot + v*w_cont_hat
#                 # if i==len(vs)-1:
#                 #     print(loc)
#                 weights1 = np.array(np.concatenate(([1.0], loc-loc_pivot)))
#                 weights2 = weights1 #because we just take the average of these and want to get this back
                
#                 xi_loc = evaluate_xi(amps, r_fine, proj_type, projfn=projfn, 
#                                 weights1=weights1, weights2=weights2, weight_type=weight_type)    
#                 xi_locs.append(xi_loc)
                
#                 p = plt.plot(r_fine, xi_loc, color=cmap(vs_norm(v)), lw=0.5)
            
#             sm = plt.cm.ScalarMappable(cmap=cmap, norm=vs_norm)
#             cbar = plt.colorbar(sm)
#             cbar.set_label(r"$v \,\, (\mathbf{x} = v\hat{e}_\mathrm{gradient} + \mathbf{x}_\mathrm{pivot})$", rotation=270, labelpad=12)
#             ax.axhline(0, color='grey', lw=0.5)
#             ax.set_ylim((-0.01, 0.12))
#             ax.set_xlabel(r"Separation $r$ ($h^{-1}\,$Mpc)")
#             ax.set_ylabel(r"$\xi(r)$")

#             ax.set_title(f"Recovered Gradient, {mock_fn}")

#             fig.savefig(os.path.join(grad_dir, f'{plots_dir}/{mock_fn}.png'))

#             plt.cla()
#             plt.close('all')


#         if prints:
#             print(f"suave --> {save_fn}")
    
#     print(f"suave --> {cat_tag}, {grad_dim}D gradient, {basis_type} basis")
#     total_time = time.time()-s
#     print(datetime.timedelta(seconds=total_time))