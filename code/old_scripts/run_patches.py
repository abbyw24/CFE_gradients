from patchify_xi import patchify, xi, xi_in_patches
from patches_lstsq_allbins import patches_lstsq_allbins
from patches_lstsq_fit import patches_lstsq_fit
from patches_gradient import patches_exp_vs_rec_vals

import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim

lognorm_file_arr = globals.lognorm_file_arr

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

n_patches = globals.n_patches

# loop through m and b arrays
for lognorm_file in lognorm_file_arr:
    for m in m_arr_perL:
        for b in b_arr:
            mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
            path_to_mocks_dir = f"mocks/{grad_dim}D/{lognorm_file}"

            # divide mock into patches and compute correlation function in each patch
            xi_in_patches(grad_dim, path_to_mocks_dir, mock_name)

            # perform a least square fit of the clustering amplitudes in each patch
            patches_lstsq_allbins(grad_dim, m, b, path_to_mocks_dir, mock_name)

            # least square fit in bin 2
            patches_lstsq_fit(grad_dim, m, b, path_to_mocks_dir, mock_name)

            # exp vs rec vals
            patches_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name)


