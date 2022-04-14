import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

import read_lognormal
from center_mock import center_mock
from xi_ls import xi_ls
from create_subdirs import create_subdirs
from suave import suave, cosmo_bases
import generate_mock_list
import globals
globals.initialize_vals()


# run Landy-Szalay on lognormal mocks directly from Kate's directory; avoid if possible
def xi_ls_ln(mock, rlz, mock_dir='/scratch/ksf293/mocks/lognormal', randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, prints=False):

    path_to_mocks_dir = os.path.join(mock_dir, mock)
    lognorm_file = f'{mock}_lognormal_rlz{rlz}.bin'

    # read in mock
    Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_mocks_dir, lognorm_file))
    L = Lx      # boxsize
    x, y, z, vx, vy, vz = data.T
    mock_data = np.array([x, y, z]).T
    center_mock(mock_data, 0, L)

    # standard Corrfunc
    # random set
    nd = len(x)
    nr = randmult*nd
    rand_set = np.random.uniform(0, L, (nr,3))
    # compute landy-szalay!
    r_avg, results_xi = xi_ls(mock_data, rand_set, periodic, nthreads, rmin, rmax, nbins)

    return r_avg, results_xi


def xi_ls_ln_mocklist(cat_tag=globals.cat_tag, randmult=globals.randmult, prints=False):

    s = time.time()
    # results for clustered mocks, NO gradient
    mock_vals = generate_mock_list.generate_mock_list(cat_tag=cat_tag, extra=True)
    lognorm_file_list = mock_vals['lognorm_file_list']
    mock_fn_list = mock_vals['mock_file_name_list']

    abs_path = '/scratch/aew492/research-summer2020_output/lognormal'
    save_dir = os.path.join(abs_path, f'xi/ls/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(lognorm_file_list)):
        xi_results = xi_ls_ln(mock_vals["lognorm_mock"], i, randmult=randmult, prints=prints)

        save_file = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn_list[i]}')
        np.save(save_file, xi_results)
        if prints:
            print(f'xi, lognormal --> {mock_fn_list[i]}')
    
    total_time = time.time()-s
    print(f'xi, lognormal {cat_tag}')
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


def xi_bao_ln_mocklist(prints=False, rmin=globals.rmin, rmax=globals.rmax, redshift=0.57, bias=2.0):

    s = time.time()
    # results for clustered mocks, NO gradient
    mock_vals = generate_mock_list.generate_mock_list(extra=True)
    lognorm_file_list = mock_vals["lognorm_file_list"]

    abs_path = '/scratch/aew492/research-summer2020_output/lognormal'
    save_dir = os.path.join(abs_path, f'xi/bao_fixed/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # use cosmo_bases to write basis file; we load this in for suave()
    projfn = '/scratch/aew492/research-summer2020_output/bases/bao_fixed/cosmo_basis.dat'
    basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)

    for i in range(len(lognorm_file_list)):
        # load data
        Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(mock_vals["path_to_lognorm_source"], f'{lognorm_file_list[i]}.bin'))
        L = Lx      # boxsize
        x, y, z, _, _, _ = data.T

        xi_results = suave(x, y, z, L, projfn)
        np.save(os.path.join(save_dir, f'xi_{lognorm_file_list[i]}'), xi_results)
        print(f'xi, suave with fixed bao --> {lognorm_file_list[i]}')
    
    total_time = time.time()-s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


def xi_bao_it_ln_mocklist(cat_tag=globals.cat_tag, prints=False):

    s = time.time()
    # results for clustered mocks, NO gradient
    mock_vals = generate_mock_list.generate_mock_list(extra=True)
    lognorm_file_list = mock_vals["lognorm_file_list"]

    abs_path = '/scratch/aew492/research-summer2020_output/lognormal'
    save_dir = os.path.join(abs_path, f'xi/bao_iterative/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(lognorm_file_list)):
        # load data
        Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(mock_vals["path_to_lognorm_source"], f'{lognorm_file_list[i]}.bin'))
        L = Lx      # boxsize
        x, y, z, _, _, _ = data.T

        projfn = os.path.join(f'/scratch/aew492/research-summer2020_output/bases/bao_iterative/tables/final_bases/basis_{cat_tag}_rlz{i}.dat')

        xi_results = suave(x, y, z, L, projfn)
        np.save(os.path.join(save_dir, f'xi_{lognorm_file_list[i]}'), xi_results)
        print(f'xi, suave with bao_iterative --> {lognorm_file_list[i]}')
    
    total_time = time.time()-s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


# lognormal catalogs have to have been run through 
def xi_ls_mocklist(cat_tag=globals.cat_tag, mock_type=globals.mock_type, boxsize=globals.boxsize, density=globals.lognormal_density,
                    prints=False, randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
                    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, data_dir=globals.data_dir, grad_dir=globals.grad_dir):

    s = time.time()

    # mocklist
    mock_fn_list = generate_mock_list.generate_mock_list(cat_tag=cat_tag)

    mock_tag = 'lognormal' if mock_type == 'lognormal' else 'gradient'

    for mock_fn in mock_fn_list:
        data_fn = os.path.join(data_dir, f'catalogs/{mock_tag}/{cat_tag}/{mock_fn}.npy')
        # mock data is in a dictionary (along w N and L), so we need to pull out just the galaxy positions
        if mock_tag == 'lognormal':
            mock_dict = np.load(data_fn, allow_pickle=True).item()
        else:
            assert mock_tag == 'gradient'
            mock_dict = np.load(data_fn, allow_pickle=True).item()
        mock_data = mock_dict['data']
        center_mock(mock_data, 0, boxsize)
        # data.shape == (N, 3)

        # random set
        random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{boxsize}_n{density}_{randmult}x.dat')
        rand_set = np.loadtxt(random_fn)
        center_mock(rand_set, 0, boxsize)

        # run landy-szalay
        r_avg, results_xi = xi_ls(mock_data, rand_set, periodic=periodic, nthreads=nthreads, rmin=rmin, rmax=rmax, nbins=nbins)

        # save directory
        if mock_tag == 'lognormal':
            save_dir = os.path.join(data_dir, f'lognormal/xi/ls/{cat_tag}')
        else:
            assert mock_tag == 'gradient'
            save_dir = os.path.join(grad_dir, f'ls/{cat_tag}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_fn = os.path.join(save_dir, f'xi_ls_{randmult}x_{mock_fn}.npy')

        np.save(save_fn, np.array([r_avg, results_xi]))

        if prints:
            print(f"landy-szalay --> {mock_fn}")
    
    total_time = time.time()-s
    print(f"landy-szalay --> {cat_tag}, {len(mock_fn_list)} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")