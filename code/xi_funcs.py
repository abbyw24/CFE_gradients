import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import Corrfunc

from center_mock import center_mock
from suave import cosmo_bases, suave, suave_grad
import random_cat
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
                    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins):
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

    # save directory
    rand_tag = '' if load_rand else '/unique_rands'

    if mock_type == 'gradient':
        save_dir = os.path.join(mock_set.grad_dir, f'ls/{cat_tag}{rand_tag}')
    else:
        assert mock_type == 'lognormal'
        save_dir = os.path.join(data_dir, f'lognormal/ls/{cat_tag}{rand_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
            try:
                rand_set = np.loadtxt(random_fn)
            except OSError: # generate the random catalog if it doesn't already exist
                random_cat.main(L, n, data_dir, randmult)
                rand_set = np.loadtxt(random_fn)
            # rand_set.shape == (nr, 3)
        else:
            nr = randmult * float(n) * int(L)**3
            rand_set = np.random.uniform(0, L, (int(nr),3))
        center_mock(rand_set, 0, L)

        # run landy-szalay
        rr_fn = os.path.join(data_dir, f'catalogs/randoms/rr_terms/rr_res_rand_L{L}_n{n}_{randmult}x.npy') if load_rand else None

        r_avg, results_xi = compute_ls(mock_data, rand_set, periodic=periodic, nthreads=nthreads, rmin=rmin, rmax=rmax, nbins=nbins, rr_fn=rr_fn)

        # results file
        save_fn = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn}.npy')
        np.save(save_fn, np.array([r_avg, results_xi]))

        if prints:
            print(f"landy-szalay --> {save_fn}")
    
    total_time = time.time()-s
    print(f"landy-szalay --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



def xi_cfe_mocklist(mock_type=globals.mock_type,
                        nmocks=globals.nmocks, L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=None,
                        prints=False, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        bao_fixed=True):
    """Use suave to estimate the continuous 2pcf on a set of mock galaxy catalogs using a specified basis."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(nmocks, L, n, As=As, data_dir=data_dir, rlzs=rlzs)
    cat_tag = mock_set.cat_tag

    # which BAO basis to use
    basis_type = 'bao_fixed' if bao_fixed else 'bao_iterative'

    # save directory (note no random catalog needed with Suave)
    if mock_type == 'gradient':
        # check whether we want to use gradient mocks or lognormal mocks
        mock_set.add_gradient(globals.grad_dim, globals.m, globals.b)   
        save_dir = os.path.join(mock_set.grad_dir, f'suave/{basis_type}/{cat_tag}')
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"
        save_dir = os.path.join(data_dir, f'lognormal/suave/{basis_type}/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # basis if bao_fixed (outside of loop because it's the same basis for all realizations)
    if bao_fixed:
        projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')
        basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)

    # run Suave on each realization in our realization list
    for i, mock_fn in enumerate(mock_set.mock_fn_list):
            
        # if bao_iterative, load in the iterative basis for this realization
        if not bao_fixed:
            projfn = os.path.join(data_dir, f'bases/bao_iterative/results/results_{mock_type}_{cat_tag}/final_bases/basis_{mock_fn}.dat')

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_type}/{cat_tag}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_mock(mock_data, 0, L)
        x, y, z = mock_data.T

        # run Suave on this data
        if mock_type=='lognormal':
            xi_results = suave(x, y, z, L, projfn)
        else:
            assert mock_type=='gradient'
            xi_results = suave_grad(x, y, z, L, projfn)

            # save suave dictionary
            save_fn = os.path.join(grad_dir, f'{suave_dir}/{mock_fn}')
            np.save(save_fn, suave_dict, allow_pickle=True)

            if prints:
                print(f"suave --> {save_fn}")
        np.save(os.path.join(save_dir, f'xi_{mock_set.ln_file_list[i]}'), xi_results)
        if prints:
            print(f'suave with {basis_type} basis --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"suave with {basis_type} basis --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")