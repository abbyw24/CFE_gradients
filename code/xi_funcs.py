import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

import read_lognormal
from center_mock import center_mock
from create_subdirs import create_subdirs
from suave import suave, cosmo_bases
import generate_mock_list
import globals
globals.initialize_vals()


def compute_ls(data, rand_set, periodic, nthreads, rmin, rmax, nbins, rr_fn=None, prints=False):
    """Run the Landy-Szalay estimator using Corrfunc"""

    # parameters
    r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    nr = len(rand_set)

    x, y, z = data.T
    x_rand, y_rand, z_rand = rand_set.T

    dd_res = Corrfunc.theory.DD(1, nthreads, r_edges, x, y, z, periodic=periodic, output_ravg=True)
    if prints == True:
        print("DD calculated")
    dr_res = Corrfunc.theory.DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, periodic=periodic)
    if prints == True:
        print("DR calculated")
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=periodic)
    if prints == True:
        print("RR calculated")

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd,nd,nr,nr,dd,dr,dr,rr)
    if prints == True:
        print("3d counts converted to cf")

    return r_avg, results_xi



def xi_ls_mocklist(mock_type=globals.mock_type,
                    nmocks=globals.nmocks, L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                    data_dir=globals.data_dir, rlzs=None,
                    prints=False, load_rand=True, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
                    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, grad_dir=globals.grad_dir):
    """Compute the Landy-Szalay 2pcf on a set of mock galaxy catalogs."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(nmocks, L, n, As=As, data_dir=data_dir, rlzs=rlzs)
    cat_tag = mock_set.cat_tag

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(globals.grad_dim, globals.m, globals.b)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    for i, mock_fn in enumerate(mock_set.mock_fn_list):
        data_fn = os.path.join(data_dir, f'catalogs/{mock_type}/{cat_tag}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        assert int(mock_dict['L']) == L, "input boxsize does not match loaded mock data!"
        center_mock(mock_data, 0, L)
        # data.shape == (N, 3)

        # random set: either load a pre-computed set, or generate one here
        if load_rand:
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{L}_n{n}_{randmult}x.dat')
            rand_set = np.loadtxt(random_fn)
            # rand_set.shape == (N, 3)
        else:
            nr = randmult * float(n) * int(L)**3
            rand_set = np.random.uniform(0, L, (int(nr),3))
        center_mock(rand_set, 0, L)

        # run landy-szalay
        rr_fn = os.path.join(data_dir, f'catalogs/randoms/rr_terms/rr_res_rand_L{L}_n{n}_{randmult}x.npy') if load_rand else None

        r_avg, results_xi = compute_ls(mock_data, rand_set, periodic=periodic, nthreads=nthreads, rmin=rmin, rmax=rmax, nbins=nbins, rr_fn=rr_fn)

        # save directory
        rand_tag = '' if load_rand else '/unique_rands'

        if mock_tag == 'gradient':
            save_dir = os.path.join(mock_set.grad_dir, f'ls/{cat_tag}{rand_tag}')
        else:
            assert mock_tag == 'lognormal'
            save_dir = os.path.join(data_dir, f'lognormal/xi/ls/{cat_tag}{rand_tag}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_fn = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn}.npy')

        np.save(save_fn, np.array([r_avg, results_xi]))

        if prints:
            print(f"landy-szalay --> {save_fn}")
    
    total_time = time.time()-s
    print(f"landy-szalay --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



# def xi_ls_ln_mocklist(cat_tag=globals.cat_tag, randmult=globals.randmult, data_dir=globals.data_dir, prints=False):

#     s = time.time()
#     # results for clustered mocks, NO gradient
#     mock_vals = generate_mock_list.generate_mock_list(cat_tag=cat_tag, extra=True)
#     lognorm_file_list = mock_vals['lognorm_file_list']
#     mock_fn_list = mock_vals['mock_file_name_list']

#     save_dir = os.path.join(data_dir, f'lognormal/xi/ls/{cat_tag}')
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     for i in range(len(lognorm_file_list)):
#         xi_results = xi_ls_ln(mock_vals["lognorm_mock"], i, randmult=randmult, prints=prints)

#         save_file = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn_list[i]}')
#         np.save(save_file, xi_results)
#         if prints:
#             print(f'xi, lognormal --> {mock_fn_list[i]}')
    
#     total_time = time.time()-s
#     print(f'xi, lognormal {cat_tag}')
#     print(f"total time: {datetime.timedelta(seconds=total_time)}")



def xi_baofix_ln_mocklist(cat_tag=globals.cat_tag, nmocks=globals.nmocks, data_dir=globals.data_dir, rlzs=None,
                            rmin=globals.rmin, rmax=globals.rmax,
                            redshift=0.57, bias=2.0, prints=False):
    "Use the CFE to estimate the 2pcf using a fixed BAO basis"

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(cat_tag, nmocks, data_dir=data_dir, rlzs=rlzs)

    # define save directory
    save_dir = os.path.join(data_dir, f'xi/bao_fixed/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # use cosmo_bases to write basis file; we load this in for suave()
    projfn = os.path.join(data_dir, 'bases/bao_fixed/cosmo_basis.dat')
    basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)

    # run Suave on each realization in our realization list
    for i, rlz in enumerate(mock_set.rlzs):

        # load lognormal data
        data_fn = os.path.join(data_dir, f'catalogs/lognormal/{cat_tag}/{mock_set.ln_file_list[i]}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_mock(mock_data, 0, L)
        x, y, z = mock_data.T

        # run Suave on this data
        xi_results = suave(x, y, z, L, projfn)
        np.save(os.path.join(save_dir, f'xi_{mock_set.ln_file_list[i]}'), xi_results)
        print(f'xi, Suave with fixed BAO basis --> {mock_set.ln_file_list[i]}')
    
    total_time = time.time()-s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


# results for clustered mocks, NO gradient
def xi_baoit_ln(cat_tag=globals.cat_tag, data_dir=globals.data_dir, prints=False):
    "Use the CFE to estimate the 2pcf using an (already computed) iterative BAO basis"

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(cat_tag, nmocks, data_dir=data_dir, rlzs=rlzs)

    # define save directory
    save_dir = os.path.join(data_dir, f'lognormal/xi/bao_iterative/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, rlz in enumerate(mock_set.rlzs):

        # load in the iterative basis for this realization
        projfn = os.path.join(data_dir, f'bases/bao_iterative/tables/final_bases/basis_{cat_tag}_rlz{rlz}.dat')

        # load lognormal data
        data_fn = os.path.join(data_dir, f'catalogs/lognormal/{cat_tag}/{mock_set.ln_file_list[i]}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_mock(mock_data, 0, L)
        x, y, z = mock_data.T

        # run Suave on this data
        xi_results = suave(x, y, z, L, projfn)
        np.save(os.path.join(save_dir, f'xi_{lognorm_file_list[i]}'), xi_results)
        print(f'xi, Suave with iterative BAO basis --> {lognorm_file_list[i]}')
    
    total_time = time.time()-s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



def xi_cfe_mocklist(cat_tag=globals.cat_tag, mock_type=globals.mock_type, mock_tag=globals.mock_tag, grad_dir=globals.grad_dir, data_dir=globals.data_dir,
                    rmin=globals.rmin, rmax=globals.rmax, bao_fixed=True, mock_range=None, prints=False):
    s = time.time()

    mock_list_info = generate_mock_list.generate_mock_list(cat_tag=cat_tag, extra=True)
    mock_file_name_list = mock_list_info['mock_file_name_list']
    mock_param_list = mock_list_info['mock_param_list']
    realizations = mock_list_info['rlz_list']

    # create the needed subdirectories
    basis_type = 'bao_fixed' if bao_fixed else 'bao_iterative'
    suave_dir = os.path.join(grad_dir, f'suave_data/{cat_tag}/{basis_type}')

    if not os.path.exists(suave_dir):
        os.makedirs(suave_dir)

    # parameters for suave (not already imported from globals)
    ncont = 2000
    r_fine = np.linspace(rmin, rmax, ncont)

    nmubins = 1
    mumax = 1.0

    # basis if bao_fixed (outside of loop because it's the same basis for all realizations)
    if bao_fixed:
        projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')
    
    mock_range = mock_range if mock_range else range(len(mock_file_name_list))

    for i in mock_range:
        rlz = realizations[i]

        mock_name = f'{cat_tag}_rlz{rlz}_lognormal' if mock_tag == 'lognormal' else f'{cat_tag}_rlz{rlz}_{mock_param_list[i]}'

        # load bases
        if not bao_fixed:
            projfn = os.path.join(data_dir, f'bases/bao_iterative/results/results_gradient_{cat_tag}/final_bases/basis_gradient_{mock_name}_trrnum_{randmult}x.dat')

        # load in mock and patch info
        mock_info = np.load(os.path.join(grad_dir, f'mock_data/{cat_tag}/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
        mock_file_name = mock_info['mock_file_name']
        L = mock_info['boxsize']
        mock_data = mock_info['grad_set']

        # center data points between 0 and L
        center_mock(mock_data, 0, L)

        nd = len(mock_data)
        x, y, z = mock_data.T

        xi_results = suave(x, y, z, L, projfn)
        suave_dict_fn = os.path.join(suave_dir, f'{mock_name}.npy')
        suave_dict = np.load(suave_dict_fn, allow_pickle=True).item()
        suave_dict['cfe_full'] = xi_results
        np.save(suave_dict_fn, suave_dict)

        if prints:
            print(f"CFE on full mock ({basis_type} basis) --> {mock_fn}")

    
    total_time = time.time()-s
    print(f'non-gradient CFE, {basis_type} basis --> {cat_tag}, {len(mock_range)} mocks')
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


# # lognormal catalogs have to have been run through 
# # **this might be buggy
# def xi_clust_mocklist(cat_tag=globals.cat_tag, mock_type=globals.mock_type, boxsize=globals.boxsize, density=globals.lognormal_density,
#                     prints=False, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
#                     rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, data_dir=globals.data_dir, grad_dir=globals.grad_dir):

#     s = time.time()

#     # mocklist
#     mock_fn_list = generate_mock_list.generate_mock_list(cat_tag=cat_tag)

#     for mock_fn in mock_fn_list:
#         data_fn = os.path.join(data_dir, f'gradient/1D/mock_data/{cat_tag}/{mock_fn}.npy')
#         # mock data is in a dictionary (along w N and L), so we need to pull out just the galaxy positions
#         mock_dict = np.load(data_fn, allow_pickle=True).item()
#         clust_data = mock_dict['clust_set']
#         L = mock_dict['boxsize']
#         center_mock(clust_data, 0, L)
#         # data.shape == (N, 3)

#         # random set
#         random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{boxsize}_n{density}_{randmult}x.dat')
#         rand_set = np.loadtxt(random_fn)
#         center_mock(rand_set, 0, boxsize)

#         # run landy-szalay
#         r_avg, results_xi = compute_ls(clust_data, rand_set, periodic=periodic, nthreads=nthreads, rmin=rmin, rmax=rmax, nbins=nbins)

#         # save directory
#         save_dir = os.path.join(grad_dir, f'ls/{cat_tag}/clustered_only')
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         save_fn = os.path.join(save_dir, f'clust_xi_ls_{randmult}x_{mock_fn}.npy')

#         np.save(save_fn, np.array([r_avg, results_xi]))

#         if prints:
#             print(f"landy-szalay, clustered only --> {mock_fn}")
    
#     total_time = time.time()-s
#     print(f"landy-szalay, clustered only --> {cat_tag}, {len(mock_fn_list)} mocks")
#     print(f"total time: {datetime.timedelta(seconds=total_time)}")