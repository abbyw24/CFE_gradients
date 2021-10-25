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

boxsize = globals.boxsize
cat_tag = globals.cat_tag


def xi_ls_ln(mock, rlz, mock_dir='/scratch/ksf293/mocks/lognormal', randmult=globals.randmult, periodic=globals.periodic, nthreads=globals.nthreads,
    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, prints=False):

    path_to_mocks_dir = os.path.join(mock_dir, mock)
    lognorm_file = f'{mock}_lognormal_rlz{rlz}.bin'

    # read in mock
    Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_mocks_dir, lognorm_file))
    L = Lx      # boxsize
    x, y, z, vx, vy, vz = data.T
        # i believe (?) the raw data is centered at L/2 (0-L)
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

def xi_ls_ln_mocklist(prints=False):

    s = time.time()
    # results for clustered mocks, NO gradient
    mock_vals = generate_mock_list.generate_mock_list(extra=True)
    lognorm_file_list = mock_vals["lognorm_file_list"]

    abs_path = '/scratch/aew492/research-summer2020_output/lognormal'
    save_dir = os.path.join(abs_path, f'xi/ls/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(lognorm_file_list)):
        xi_results = xi_ls_ln(mock_vals["lognorm_mock"], i, prints=prints)
        np.save(os.path.join(save_dir, f'xi_{lognorm_file_list[i]}'), xi_results)
        print(f'xi, {lognorm_file_list[i]}')
    
    total_time = time.time()-s
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

# currently bao_iterative.py calculates this, so you shouldn't need to call this function; it's redundant
def xi_bao_it_ln_mocklist(prints=False):

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