from suave_gradient import cosmo_bases, suave_exp_vs_rec_vals

import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim

lognorm_file_arr = globals.lognorm_file_arr

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

# loop through m and b arrays
for lognorm_file in lognorm_file_arr:
    for m in m_arr_perL:
        for b in b_arr:
            mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
            path_to_mocks_dir = f"mocks/{grad_dim}D/{lognorm_file}"
            suave_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name)