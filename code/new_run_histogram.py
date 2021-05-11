import numpy as np
import matplotlib.pyplot as plt
from histogram import histogram
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
L = globals.L
loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_patches = globals.n_patches

def patches(method):
    if method == "patches":
        return f"_{n_patches}patches"
    else:
        return ""

## recovered gradient should have the form (3, nrealizations)

method = ["suave", "patches"]

grads_recovered_dict = {}

for j in method:
    # create list to add the recovered gradients (shape (3,)) from each realization
    grads_recovered = []
    print(f"{hist_type} histogram, {j}")
    for m in m_arr_perL:
        for b in b_arr:
            data = np.load(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/exp_vs_rec_vals/patches_exp_vs_rec_{n_patches}patches_{mock_name}{patches(j)}.npy"), allow_pickle=True).item()
            grads_recovered.append(data["grad_recovered"])

    # add recovered gradients from this method to a dictionary entry
    grads_recovered_dict[j] = np.array(grads_recovered)

patches_data = grads_recovered_dict["patches"]
suave_data = grads_recovered_dict["suave"]

# histogram(patches_data, suave_data, 