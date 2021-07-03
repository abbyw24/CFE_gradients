import numpy as np
import matplotlib.pyplot as plt
import os
import read_lognormal
from center_mock import center_mock
from corrfunc_ls import xi_ls
from create_subdirs import create_subdirs
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

sub_dirs = [
    'xi'
]
abs_path = '/scratch/aew492/research-summer2020_output/lognormal'
create_subdirs(abs_path, sub_dirs)

mock = f'cat_L{globals.boxsize}_n{globals.lognormal_density}_z057_patchy'

for i in range(len(globals.mock_file_name_list)):
    r_avg, results_xi = xi_lognormal.xi_lognormal(mock, i)
    np.save(os.path.join(abs_path, f'xi/xi_{globals.mock_file_name_list[i]}'), results_xi)
    print(f'mock {i} done')

