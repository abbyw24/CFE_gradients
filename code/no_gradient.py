import numpy as np
import matplotlib.pyplot as plt
import os
import read_lognormal
from center_mock import center_mock
from corrfunc_ls import xi_ls
import globals
globals.initialize_vals()

# OUTLINE
# loop:
#   pull up mock from /scratch/ksf293/mocks/lognormal
#   * NO gradient injection– use 'plain' clustered mock
#   run standard Corrfunc (same process as calculating xi_full)

mock_dir = '/scratch/ksf293/mocks/lognormal'
mock = 'cat_L750_n2e-4_z057_patchy'

path_to_mocks_dir = os.path.join(mock_dir, mock)

# the following should eventually be in the loop
lognorm_file = 'cat_L750_n2e-4_z057_patchy_lognormal_rlz0.bin'
# read in mock
Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_mocks_dir, lognorm_file))
L = Lx      # boxsize
x, y, z, vx, vy, vz = data.T
    # i believe (?) the raw data is centered at L/2 (0-L)
mock_data = np.array([x, y, z]).T
print(mock_data.shape)
center_mock(mock_data, 0, L)

# standard Corrfunc
# random set
nd = len(x)
nr = 2*nd
rand_set = np.random.uniform(0, L, (nr,3))

# compute landy-szalay!
results_xi = xi_ls(mock_data, rand_set, globals.periodic, globals.nthreads, globals.rmin, globals.rmax, globals.nbins)
print(results_xi)