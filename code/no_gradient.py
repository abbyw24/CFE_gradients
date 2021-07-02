import numpy as np
import matplotlib.pyplot as plt
import os
import read_lognormal
from center_mock import center_mock
from corrfunc_ls import xi_ls
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

xi_lognormal('cat_L750_n2e-4_z057_patchy', 0)