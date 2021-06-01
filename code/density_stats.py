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

n_patches = globals.n_patches

def histogram_patches_vs_suave(densities, method, grad_type=grad_type, path_to_data_dir=path_to_data_dir, n_patches=n_patches, nbins=10):
    # make sure inputs have correct form
    assert method == "patches" or "suave"
    assert isinstance(densities, list)

    # create the needed subdirectories
    sub_dirs = [
        f"plots/densities/histogram/{grad_type}/{n_mocks}mocks"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)
    
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # get recovered gradients
    grads_rec = {}

    if method == "patches":
        for n in densities:
            grads_rec[n] = []
            for j in range(len(mock_file_name_list)):
                info = np.load(os.path.join(path_to_data_dir, f"patch_data/{n}/{n_patches}patches/{mock_file_name_list[j]}.npy"), allow_pickle=True).item()
                grad_rec = info["grad_recovered"]
                grads_rec[n].append(grad_rec)
    elif method == "suave":
        for n in densities:
            grads_rec[n] = []
            for j in range(len(mock_file_name_list)):
                info = np.load(os.path.join(path_to_data_dir, f"suave_data/{n}/{mock_file_name_list[j]}.npy"), allow_pickle=True).item()
                grad_rec = info["grad_recovered"]
                grads_rec[n].append(grad_rec)
    else:
        print("method must be either 'patches' or 'suave'")
        assert False

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {densities}, {dim[i]}, {grad_type}, {n_mocks} mocks")
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