import numpy as np
import matplotlib.pyplot as plt
import globals

from histogram import histogram

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

n_sides = globals.n_sides
n_patches = n_sides**3

def patches(method):
    if method == "patches":
        return f"_{n_patches}patches"
    else:
        return ""

## recovered gradient should have the form (3, nrealizations)

### specify hist_type out of "yz" or "null-downsampled"
hist_type = "yz"
###

method = ["suave", "patches"]

grads_recovered_dict = {}

for j in method:
    # create list to add the recovered gradients (shape (3,)) from each realization
    grads_recovered = []
    if hist_type == "yz":
        for m in m_arr_perL:
            for b in b_arr:
                data = np.load(f"gradient_mocks/{grad_dim}D/{j}/exp_vs_rec_vals/{j}_exp_vs_rec_vals_m-{m}-L_b-{b}{patches(j)}.npy", allow_pickle=True).item()
                grads_recovered.append(data["grad_recovered"])
        dim = {
            1 : "y",
            2 : "z"
            }
    elif hist_type == "null-full":
        dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }
    else:
        print("hist_type must be either 'yz' or 'null-full'")

    # add recovered gradients from this method to a dictionary entry
    grads_recovered_dict[j] = np.array(grads_recovered)

histogram(hist_type="yz", grads_recovered_dict["patches"], grads_recovered_dict["suave"], dim=dim)