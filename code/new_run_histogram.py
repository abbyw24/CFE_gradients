import numpy as np
import matplotlib.pyplot as plt
import os

from histogram import histogram
from histogram import hist_patches_vs_suave_1rlz
import globals

globals.initialize_vals()  # brings in all the default parameters

path_to_mocks_dir = globals.path_to_mocks_dir

grad_dim = globals.grad_dim
loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_patches = globals.n_patches

def method_path(method, mock_name):
    if method == "patches":
        return f"patches/lst_sq_fit/exp_vs_rec_vals/patches_exp_vs_rec_{n_patches}patches_{mock_name}.npy"
    elif method == "suave":
        return f"suave/recovered/exp_vs_rec_vals/suave_exp_vs_rec_{mock_name}.npy"
    else:
        return "'method' must be either 'patches' or 'suave'"

## recovered gradient should have the form (3, nrealizations)

# pull out recovered gradients that we need for the histogram
method = ["suave", "patches"]
grads_recovered_dict = {}

for j in method:
    # create list to add the recovered gradients (shape (3,)) from each realization
    grads_recovered = []
    print(f"histogram, {j}")
    for m in m_arr_perL:
        for b in b_arr:
            mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
            data = np.load(os.path.join(path_to_mocks_dir, method_path(j, mock_name)), allow_pickle=True).item()
            grads_recovered.append(data["grad_recovered"])
    # add recovered gradients from this method to a dictionary entry
    grads_recovered_dict[j] = np.array(grads_recovered)

patches_data = grads_recovered_dict["patches"]
suave_data = grads_recovered_dict["suave"]

hist_patches_vs_suave_1rlz(patches_data, suave_data, path_to_mocks_dir)