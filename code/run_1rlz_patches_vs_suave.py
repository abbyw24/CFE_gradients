import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
from gradmock_gen import generate_gradmock
from patchify_xi import xi_in_patches
from patches_lstsq_allbins import patches_lstsq_allbins
from patches_lstsq_fit import patches_lstsq_fit
from suave_gradient import suave_exp_vs_rec_vals
from exp_vs_rec_scatter import scatter_patches_vs_suave_1rlz
from exp_vs_rec_histogram import hist_patches_vs_suave_1rlz

import globals

globals.initialize_vals()

grad_dim = globals.grad_dim

lognorm_file = globals.lognorm_file
path_to_mocks_dir = globals.path_to_mocks_dir
path_to_lognorm_source = globals.path_to_lognorm_source

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

for m in m_arr_perL:
    for b in b_arr:
        mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)

        # generate grad mocks from specified lognorm file, and m and b arrays
        generate_gradmock(grad_dim, m, b, path_to_lognorm_source, lognorm_file, path_to_mocks_dir, mock_name)

        # PATCHES
        # divide mock into patches and compute correlation function in each patch
        xi_in_patches(grad_dim, path_to_mocks_dir, mock_name)

        # perform a least square fit of the clustering amplitudes in each patch
        patches_lstsq_allbins(grad_dim, m, b, path_to_mocks_dir, mock_name)

        # least square fit in bin 2
        patches_lstsq_fit(grad_dim, m, b, path_to_mocks_dir, mock_name)

        # SUAVE
        suave_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name)

# PATCHES VS SUAVE, EXPECTED VS RECOVERED
scatter_patches_vs_suave_1rlz(mock_name, path_to_mocks_dir)
hist_patches_vs_suave_1rlz(mock_name, path_to_mocks_dir)