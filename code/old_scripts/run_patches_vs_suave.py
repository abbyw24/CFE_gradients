import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
from gradmock_gen import generate_gradmock
from patchify_xi import xi_in_patches
from patches_lstsq_allbins import patches_lstsq_allbins
from patches_lstsq_fit import patches_lstsq_fit
from patches_gradient import patches_exp_vs_rec_vals
from suave_gradient import suave_exp_vs_rec_vals
from extract_grads import extract_grads_exp_vs_rec
from exp_vs_rec_scatter import scatter_exp_vs_rec
from exp_vs_rec_histogram import histogram_exp_vs_rec

from create_subdirs import create_subdirs

import globals

globals.initialize_vals()

grad_dim = globals.grad_dim

path_to_lognorm_source = globals.path_to_lognorm_source
lognorm_file_arr = globals.lognorm_file_arr

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

grad_type = globals.grad_type

# for lognorm_file in lognorm_file_arr:
#     for m in m_arr_perL:
#         for b in b_arr:
#             mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
#             path_to_mocks_dir = f"mocks/{grad_dim}D/{lognorm_file}"

#             # generate grad mocks from specified lognorm file, and m and b arrays
#             generate_gradmock(grad_dim, m, b, path_to_lognorm_source, lognorm_file, path_to_mocks_dir, mock_name)

#             # PATCHES
#             # divide mock into patches and compute correlation function in each patch
#             xi_in_patches(grad_dim, path_to_mocks_dir, mock_name)

#             # perform a least square fit of the clustering amplitudes in each patch
#             patches_lstsq_allbins(grad_dim, m, b, path_to_mocks_dir, mock_name)

#             # least square fit in bin 2
#             patches_lstsq_fit(grad_dim, m, b, path_to_mocks_dir, mock_name)

#             # exp vs rec vals
#             patches_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name)

#             # SUAVE
#             suave_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name)

#             print (" ") # for nice loop print formatting

# extract recovered and expected values for patches and suave
grads = extract_grads_exp_vs_rec()

# PATCHES VS SUAVE, EXPECTED VS RECOVERED
# scatter
create_subdirs(f"scratch/aew492/{grad_dim}D", ["plots/scatter"])
path_to_scatter_dir = f"/scratch/aew492/{grad_dim}D/plots/scatter"
scatter_exp_vs_rec(grads["grads_exp_p"], grads["grads_rec_p"], grads["grads_exp_s"], grads["grads_rec_s"], path_to_scatter_dir)

# histogram
create_subdirs(f"scratch/aew492/{grad_dim}D", ["plots/histograms"])
path_to_hist_dir = f"/scratch/aew492/{grad_dim}D/plots/histograms"
histogram_exp_vs_rec(grads["grads_rec_p"], grads["grads_rec_s"], path_to_hist_dir)