import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

from funcs.center_data import center_data
import funcs.random_cat
from funcs.corrfunc_ls import compute_ls
from funcs.suave import cosmo_bases

import generate_mock_list
import fetch_lognormal_mocks
from suave_mocks import suave, suave_grad
from patches_lstsq_fit import compute_patches_lstsqfit
import globals
globals.initialize_vals()


# XI SCRIPTS: Estimate the two-point correlation function (i.e. no gradient estimation) in a mock galaxy catalog

def xi_ls_mocklist(mock_type=globals.mock_type,
                    L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                    data_dir=globals.data_dir, rlzs=globals.rlzs,
                    grad_dim=globals.grad_dim, m=globals.m, b=globals.b, same_dir=globals.same_dir,
                    prints=False, load_rand=True, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
                    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, overwrite=False):
    """Compute the Landy-Szalay 2pcf on a set of mock galaxy catalogs."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.MockSet(L, n, As=As, data_dir=data_dir, rlzs=rlzs)

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b, same_dir=same_dir)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/ls')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, mock_fn in enumerate(mock_set.mock_fn_list):
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{mock_fn}.npy')
        try:
            mock_dict = np.load(data_fn, allow_pickle=True).item()
        # if running on pure clustered mocks, try fetching them from /ksf293 if they aren't already in the catalog directory
        except FileNotFoundError:
            if mock_type=='lognormal':
                print("Fetching lognormal catalogs...")
                fetch_lognormal_mocks.fetch_ln_mocks(mock_set.cat_tag, mock_set.rlzs)
                mock_dict = np.load(data_fn, allow_pickle=True).item()
            else:
                assert FileNotFoundError, f"cannot find {data_fn}"

        mock_data = mock_dict['data']
        assert int(mock_dict['L']) == L, "input boxsize does not match loaded mock data!"
        center_data(mock_data, 0, L)
        # data.shape == (N, 3)

        save_fn = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn}.npy')

        # check if we've already calculated for this mock
        if not overwrite:
            if os.path.exists(save_fn):
                print(f'L-S already computed for mock {mock_fn}! moving to the next.')
                continue

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
        center_data(rand_set, 0, L)

        # run landy-szalay
        rr_fn = os.path.join(data_dir, f'catalogs/randoms/rr_terms/rr_res_rand_L{L}_n{n}_{randmult}x.npy') if load_rand else None

        r_avg, xi = compute_ls(mock_data, rand_set, periodic=periodic, nthreads=nthreads, rmin=rmin, rmax=rmax, nbins=nbins, rr_fn=rr_fn)

        # results file
        
        np.save(save_fn, np.array([r_avg, xi]))

        if prints:
            print(f"landy-szalay --> {save_fn}")
    
    total_time = time.time()-s
    print(f"landy-szalay --> {save_dir}, {mock_set.nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



def xi_cfe_mocklist(mock_type=globals.mock_type,
                        L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=globals.rlzs,
                        grad_dim=globals.grad_dim, m=globals.m, b=globals.b, same_dir=globals.same_dir,
                        prints=False, load_rand=True, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        basis='bao_iterative', overwrite=False):
    """Use Suave to estimate the continuous 2pcf on a set of mock galaxy catalogs using a specified basis."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.MockSet(L, n, As=As, data_dir=data_dir, rlzs=rlzs)

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b, same_dir=same_dir)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/suave/xi/{basis}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # basis if bao_fixed (outside of loop because it's the same basis for all realizations)
    if basis!='bao_iterative':
        projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')
        basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)

    # run Suave on each realization in our realization list
    for i, mock_fn in enumerate(mock_set.mock_fn_list):
            
        # if bao_iterative, load in the iterative basis for this realization
        if basis=='bao_iterative':
            projfn = os.path.join(data_dir, f'bases/bao_iterative/{mock_set.mock_path}/results/final_bases/basis_{mock_fn}_trrnum_{randmult}x.dat')
            if not os.path.exists(projfn):
                assert FileNotFoundError, "iterative basis not found!"

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_data(mock_data, 0, L)
        x, y, z = mock_data.T

        save_fn = os.path.join(save_dir, f'xi_{mock_fn}.npy')

        # check if we've already calculated for this mock
        if not overwrite:
            if os.path.exists(save_fn):
                print(f'CFE already computed for mock {mock_fn}! moving to the next.')
                continue

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

        # run Suave on this data
        xi_results = suave(x, y, z, L, n, projfn, load_rand=load_rand)

        np.save(save_fn, xi_results)

        if prints:
            print(f'suave with {basis} basis --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"suave with {basis} basis --> {save_dir}, {mock_set.nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")



# GRADIENT SCRIPTS: Estimate the clustering gradient in a mock galaxy catalog

def grad_cfe_mocklist(mock_type=globals.mock_type,
                        L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=globals.rlzs,
                        grad_dim=globals.grad_dim, m=globals.m, b=globals.b, same_dir=globals.same_dir,
                        prints=False, load_rand=True, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        basis='bao_iterative', overwrite=False):
    """Use Suave to estimate the clustering gradients on a set of mock galaxy catalogs using a specified basis."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.MockSet(L, n, As=As, data_dir=data_dir, rlzs=rlzs)

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b, same_dir=same_dir)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory (note no random catalog needed with Suave)
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/suave/grad_amps/{basis}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # basis if bao_fixed (outside of loop because it's the same basis for all realizations)
    if basis!='bao_iterative':
        projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')

    # run Suave on each realization in our realization list
    for i, mock_fn in enumerate(mock_set.mock_fn_list):
            
        # if bao_iterative, load in the iterative basis for this realization
        if basis=='bao_iterative':
            projfn = os.path.join(data_dir, f'bases/bao_iterative/{mock_set.mock_path}/results/final_bases/basis_{mock_fn}_trrnum_{randmult}x.dat')
            if not os.path.exists(projfn):
                assert FileNotFoundError, "iterative basis not found!"

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_data(mock_data, 0, L)
        x, y, z = mock_data.T

        save_fn = os.path.join(save_dir, f'{mock_fn}.npy')

        # check if we've already calculated for this mock
        if not overwrite:
            if os.path.exists(save_fn):
                print(f'CFE grad already computed for mock {mock_fn}! moving to the next.')
                continue

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

        # run Suave on this data
        results_dict = suave_grad(x, y, z, L, n, projfn, load_rand=load_rand)

        np.save(save_fn, results_dict)

        if prints:
            print(f'suave_grad with {basis} basis --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"suave_grad with {basis} basis --> {save_dir}, {mock_set.nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")
    


def grad_patches_mocklist(mock_type=globals.mock_type,
                        L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
                        data_dir=globals.data_dir, rlzs=globals.rlzs,
                        grad_dim=globals.grad_dim, m=globals.m, b=globals.b, same_dir=globals.same_dir,
                        prints=False, load_rand=True, periodic=globals.periodic, nthreads=globals.nthreads,
                        rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
                        npatches=globals.npatches, overwrite=False):
    """Use a standard approach to estimate the clustering gradients on a set of mock galaxy catalogs."""

    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.MockSet(L, n, As=As, data_dir=data_dir, rlzs=rlzs)

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b, same_dir=same_dir)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory (note no random catalog needed with Suave)
    save_dir = os.path.join(data_dir, f'{mock_set.mock_path}/patches/{npatches}patches/grad_amps')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # run patchify on each realization in our realization list
    for i, mock_fn in enumerate(mock_set.mock_fn_list):

        # first check if the basis file exists
        basis_fn = os.path.join(data_dir, f'bases/4-parameter_fit/scipy/{mock_set.mock_path}/basis_{mock_fn}.npy')
        assert os.path.exists(basis_fn), "could not find basis file from 4-parameter scipy fit!"

        # load data
        data_fn = os.path.join(data_dir, f'catalogs/{mock_set.mock_path}/{mock_fn}.npy')
        mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        L = mock_dict['L']
        center_data(mock_data, 0, L)
        x, y, z = mock_data.T

        save_fn = os.path.join(save_dir, f'{mock_fn}.npy')

        # check if we've already calculated for this mock
        if not overwrite:
            if os.path.exists(save_fn):
                print(f'patches grad already computed for mock {mock_fn}! moving to the next.')
                continue

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_fn)

        # run patches method on this data: compute xi in patches, and perform the least-squares fit
        results_dict = compute_patches_lstsqfit(x, y, z, L, n, basis_fn)

        np.save(save_fn, results_dict)

        if prints:
            print(f'xi in patches --> {mock_fn}')
    
    total_time = time.time()-s
    print(f"patches method --> {save_dir}, {mock_set.nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")