import numpy as np
import matplotlib.pyplot as plt
import os

import globals
from create_subdirs import create_subdirs
from extract_grads import extract_grads_exp_vs_rec

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim

lognorm_file_arr = globals.lognorm_file_arr

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

grad_type = globals.grad_type

n_patches = globals.n_patches

def label_s(k):
    if k == 0:
        return "suave"
    else:
        return None

def label_p(k):
    if k == 0:
        return "patches"
    else:
        return None

def scatter_exp_vs_rec(grads_exp_patches, grads_rec_patches, grads_exp_suave, grads_rec_suave, path_to_scatter_dir):
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }
    k = 0

    # make sure all inputs have the right form
    assert grads_exp_patches.shape == grads_exp_suave.shape == grads_rec_patches.shape == grads_rec_suave.shape
    # grads.shape == (201, 3)
    assert isinstance(path_to_scatter_dir, str)

    for i in dim:
        # create plot
        fig, ax = plt.subplots()
        ax.set_xlabel("Expected Gradient")
        ax.set_ylabel("Recovered Gradient")
        ax.set_title(f"Expected vs. Recovered Gradient, {dim[i]}")

        for j in range(len(grads_exp_patches[:,i])):
            plt.plot(grads_exp_patches[j,i], grads_rec_patches[j,i], marker=".", color="C0", alpha=0.5, label=label_p(k))
            plt.plot(grads_exp_suave[j,i], grads_rec_suave[j,i], marker=".", color="orange", alpha=0.5, label=label_s(k))
            k += 1
        
        # plot line y = x (the data points would fall on this line if the expected and recovered gradients matched up perfectly)
        x = np.linspace(min(grads_exp_patches[:,i]), max(grads_exp_patches[:,i]), 10)
        plt.plot(x, x, color="black", alpha=0.5)
        plt.legend()

        # if grad_type == "1rlz":
        #     sub_dirs = ["plots/scatter"]
        #     create_subdirs(path_to_mocks_dir, sub_dirs)
        #     path_to_scatter_dir = f"mocks/{grad_dim}D/plots/scatter")
        # elif grad_type == "1m":
        #     sub_dirs = ["plots/scatter"]
        #     create_subdirs(f"mocks/{grad_dim}D", sub_dirs)
        #     path_to_scatter_dir = os.path.join()
        # else:
        #     print("'grad_type' must be '1rlz' or '1m'")
        
        fig.savefig(os.path.join(path_to_scatter_dir, f"scatter_patches_vs_suave_{dim[i]}.png"))
        plt.cla()