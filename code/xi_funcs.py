import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

import generate_mock_list
from center_mock import center_mock
import random_cat
from corrfunc_ls import compute_ls
from suave import cosmo_bases, suave, suave_grad
import globals
globals.initialize_vals()


# XI SCRIPTS: Estimate the two-point correlation function (i.e. no gradient estimation) in a mock galaxy catalog

def xi_ls_mocklist(mock_type=globals.mock_type,
                    L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                    data_dir=globals.data_dir, rlzs=None, nmocks=globals.nmocks, 
                    grad_dim=globals.grad_dim, m=globals.m, b=globals.b,
                    prints=False, load_rand=True, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
                    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins):
    """Compute the Landy-Szalay 2pcf on a set of mock galaxy catalogs."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(L, n, As=As, data_dir=data_dir, rlzs=rlzs, nmocks=nmocks)
    cat_tag = mock_set.cat_tag

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory
    rand_tag = '' if load_rand else '/unique_rands'
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/ls/{cat_tag}{rand_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, mock_fn in enumerate(mock_set.mock_fn_list):
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{cat_tag}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        assert int(mock_dict['L']) == L, "input boxsize does not match loaded mock data!"
        center_mock(mock_data, 0, L)
        # data.shape == (N, 3)

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

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

        r_avg, xi = compute_ls(mock_data, rand_set, periodic=periodic, nthreads=nthreads, rmin=rmin, rmax=rmax, nbins=nbins, rr_fn=rr_fn)

        # results file
        save_fn = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn}.npy')
        np.save(save_fn, np.array([r_avg, xi]))

        if prints:
            print(f"landy-szalay --> {save_fn}")
    
    total_time = time.time()-s
    print(f"landy-szalay --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



def xi_cfe_mocklist(mock_type=globals.mock_type,
                        L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=None, nmocks=globals.nmocks, 
                        grad_dim=globals.grad_dim, m=globals.m, b=globals.b,
                        prints=False, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        bao_fixed=True):
    """Use suave to estimate the continuous 2pcf on a set of mock galaxy catalogs using a specified basis."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(L, n, As=As, data_dir=data_dir, rlzs=rlzs, nmocks=nmocks)
    cat_tag = mock_set.cat_tag

    # which BAO basis to use
    basis_type = 'bao_fixed' if bao_fixed else 'bao_iterative'

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory (note no random catalog needed with Suave)
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/suave/xi/{basis_type}/{cat_tag}')
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
            if not os.file.exists(projfn):
                assert FileNotFoundError, "iterative basis not found!"

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{cat_tag}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_mock(mock_data, 0, L)
        x, y, z = mock_data.T

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

        # run Suave on this data
        xi_results = suave(x, y, z, L, n, projfn)

        np.save(os.path.join(save_dir, f'xi_{mock_fn}'), xi_results)

        if prints:
            print(f'suave with {basis_type} basis --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"suave with {basis_type} basis --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



# GRADIENT SCRIPTS: Estimate the clustering gradient in a mock galaxy catalog

def grad_cfe_mocklist(mock_type=globals.mock_type,
                        L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=None, nmocks=globals.nmocks, 
                        grad_dim=globals.grad_dim, m=globals.m, b=globals.b,
                        prints=False, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        bao_fixed=True):
    """Use suave to estimate the clustering gradients on a set of mock galaxy catalogs using a specified basis."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(L, n, As=As, data_dir=data_dir, rlzs=rlzs, nmocks=nmocks)
    cat_tag = mock_set.cat_tag

    # which BAO basis to use
    basis_type = 'bao_fixed' if bao_fixed else 'bao_iterative'

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory (note no random catalog needed with Suave)
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/suave/grad_amps/{basis_type}/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # basis if bao_fixed (outside of loop because it's the same basis for all realizations)
    if bao_fixed:
        projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')
        basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)

    # run Suave on each realization in our realization list
    for i, mock_fn in enumerate(mock_set.ock_fn_list):
            
        # if bao_iterative, load in the iterative basis for this realization
        if not bao_fixed:
            projfn = os.path.join(data_dir, f'bases/bao_iterative/results/results_{mock_type}_{cat_tag}/final_bases/basis_{mock_fn}.dat')
            if not os.file.exists(projfn):
                assert FileNotFoundError, "iterative basis not found!"

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{cat_tag}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_mock(mock_data, 0, L)
        x, y, z = mock_data.T

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

        # run Suave on this data
        results_dict = suave_grad(x, y, z, L, projfn)

        np.save(os.path.join(save_dir, mock_fn), results_dict)

        if prints:
            print(f'suave_grad with {basis_type} basis --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"suave_grad with {basis_type} basis --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")
    


def grad_patches_mocklist(mock_type=globals.mock_type,
                        L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=None, nmocks=globals.nmocks, 
                        grad_dim=globals.grad_dim, m=globals.m, b=globals.b,
                        prints=False, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        bao_fixed=True):
    """Use a standard approach to estimate the clustering gradients on a set of mock galaxy catalogs."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.mock_set(L, n, As=As, data_dir=data_dir, rlzs=rlzs, nmocks=nmocks)
    cat_tag = mock_set.cat_tag

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory (note no random catalog needed with Suave)
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/patches/grad_amps/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # run Suave on each realization in our realization list
    for i, mock_fn in enumerate(mock_set.mock_fn_list):

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{cat_tag}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_mock(mock_data, 0, L)
        x, y, z = mock_data.T

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

        # run Suave on this data
        results_dict = suave_grad(x, y, z, L, projfn)

        np.save(os.path.join(save_dir, mock_fn), results_dict)

        if prints:
            print(f'suave_grad with {basis_type} basis --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"suave_grad with {basis_type} basis --> {save_dir}, {nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")