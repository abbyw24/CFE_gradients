import numpy as np
import matplotlib.pyplot as plt
import os

import globals
from create_subdirs import create_subdirs

globals.initialize_vals()  # brings in all the default parameters

path_to_mocks_dir = globals.path_to_mocks_dir

grad_dim = globals.grad_dim
loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

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

def scatter_exp_vs_rec(grads_exp_patches, grads_rec_patches, grads_exp_suave, grads_rec_suave, path_to_mocks_dir):
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }
    k = 0

    # create desired path to mocks directory if it doesn't already exist
    sub_dirs = ["plots/scatter"]
    create_subdirs(path_to_mocks_dir, sub_dirs)
    path_to_scatter_dir = os.path.join(path_to_mocks_dir, f"plots/scatter")
    scatter_name = f"scatter_patches_vs_suave"

    # make sure all inputs have the right form
    assert grads_exp_patches.shape == grads_exp_suave.shape == grads_rec_patches.shape == grads_rec_suave.shape
    # grads.shape == (201, 3)
    assert isinstance(path_to_mocks_dir, str)

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
        
        fig.savefig(os.path.join(path_to_scatter_dir, f"{scatter_name}_{dim[i]}.png"))
        plt.cla()

def scatter_patches_vs_suave_1rlz(mock_name, path_to_mocks_dir):
    grads_exp_p = []
    grads_rec_p = []
    grads_exp_s = []
    grads_rec_s = []
    # loop through m and b values
    for m in m_arr_perL:
        for b in b_arr:
            # mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
            patches_data = np.load(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/exp_vs_rec_vals/patches_exp_vs_rec_{n_patches}patches_{mock_name}.npy"), allow_pickle=True).item()
            grad_exp_p = patches_data["grad_expected"]
            grad_rec_p = patches_data["grad_recovered"]
            grads_exp_p.append(grad_exp_p)
            grads_rec_p.append(grad_rec_p)
            
            suave_data = np.load(os.path.join(path_to_mocks_dir, f"suave/recovered/exp_vs_rec_vals/suave_exp_vs_rec_{mock_name}.npy"), allow_pickle=True).item()
            grad_exp_s = suave_data["grad_expected"]
            grad_rec_s = suave_data["grad_recovered"]
            grads_exp_s.append(grad_exp_s)
            grads_rec_s.append(grad_rec_s)
    grads_exp_p = np.array(grads_exp_p)
    grads_rec_p = np.array(grads_rec_p)
    grads_exp_s = np.array(grads_exp_s)
    grads_rec_s = np.array(grads_rec_s)

    scatter_exp_vs_rec(grads_exp_p, grads_rec_p, grads_exp_s, grads_rec_s, path_to_mocks_dir)