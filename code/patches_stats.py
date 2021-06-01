import numpy as np
import matplotlib.pyplot as plt
import os

import globals
from create_subdirs import create_subdirs

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
path_to_data_dir = globals.path_to_data_dir
mock_file_name_list = globals.mock_file_name_list
n_mocks = globals.n_mocks

grad_type = globals.grad_type

lognormal_density = globals.lognormal_density

def histogram_patches_vs_suave(n_patches_list, grad_type=grad_type, lognormal_density=lognormal_density, path_to_data_dir=path_to_data_dir, nbins=10):
    # make sure inputs have correct form
    assert isinstance(n_patches_list, list)

    # create the needed subdirectories
    sub_dirs = [
        f"plots/n_patches/histogram/{grad_type}/{n_mocks}mocks"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)
    
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # get recovered gradients
    grads_rec = {}

    for n_patches in n_patches_list:
        grads_rec[str(n_patches)] = []
        for j in range(len(mock_file_name_list)):
            info = np.load(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name_list[j]}.npy"), allow_pickle=True).item()
            grad_rec = info["grad_recovered"]
            grads_rec[str(n_patches)].append(grad_rec)

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {n_patches_list}, {dim[i]}, {grad_type}, {n_mocks} mocks")
        plt.xlabel("Recovered Gradient")

        # define bins
        bins = np.linspace(1.5*min(grads_rec), 1.5*max(grads_rec), nbins)
        assert False
        n_s, _, _ = plt.hist(grads_rec_suave[:,i], bins=bins, color="indigo", alpha=0.6, label="CFE")
        n_p, _, _ = plt.hist(grads_rec_patches[:,i], bins=bins, color="gray", alpha=0.6, label="Standard", zorder=100)

        # line at x = 0
        plt.vlines(0, 0, max(max(n_s), max(n_p)), color="black", alpha=1, zorder=101, linewidth=1)

        plt.legend()

        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/histogram/{lognormal_density}/{grad_type}/{n_mocks}mocks/hist_{n_patches}patches_vs_suave_{nbins}bins_{dim[i]}.png"))
        plt.cla()

        print(f"histogram for patches vs. suave, dim {dim[i]}, done")