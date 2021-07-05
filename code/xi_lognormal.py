import numpy as np
import matplotlib.pyplot as plt
import os
import read_lognormal
from center_mock import center_mock
from corrfunc_ls import xi_ls
from create_subdirs import create_subdirs
import generate_mock_list
import globals
globals.initialize_vals()

def xi_lognormal(mock, rlz, mock_dir='/scratch/ksf293/mocks/lognormal', randmult=2, periodic=globals.periodic, nthreads=globals.nthreads,
    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins):

    path_to_mocks_dir = os.path.join(mock_dir, mock)
    lognorm_file = str(mock)+'_lognormal_rlz'+str(rlz)+'.bin'

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


# results for clustered mocks, NO gradient
mock_vals = generate_mock_list.generate_mock_list(As=1, extra=True)
mock_file_name_list = mock_vals["mock_file_name_list"]

sub_dirs = [
    'xi'
]
abs_path = '/scratch/aew492/research-summer2020_output/lognormal'
create_subdirs(abs_path, sub_dirs)

for i in range(len(mock_file_name_list)):
    xi_results = xi_lognormal(mock_vals["lognorm_mock"], i)
    np.save(os.path.join(abs_path, f'xi/xi_{mock_file_name_list[i]}'), xi_results)
    print(f'xi, {mock_file_name_list[i]}')

