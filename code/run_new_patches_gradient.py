from new_patches_gradient import patches_exp_vs_rec_vals

import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim

lognorm_file = globals.lognorm_file
path_to_mocks_dir = globals.path_to_mocks_dir

loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_patches = globals.n_patches

# loop through m and b arrays
for m in m_arr_perL:
    for b in b_arr:
        mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
        patches_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name)