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

def histogram_patches(n_patches_list, grad_type=grad_type, lognormal_density=lognormal_density, path_to_data_dir=path_to_data_dir, nbins=20):
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
        a = 0.8
        bin_vals = []
        for n_patches in n_patches_list:
            grads_rec = grads_rec[str(n_patches)]
            n, _, _ = plt.hist(grads_rec[:,i], bins=bins, color="indigo", alpha=a, label=f"{n_patches} patches")
            a /= 2
            bin_vals.append(n)

        bin_vals = np.array(bin_vals)

        # line at x = 0
        plt.vlines(0, 0, np.amax(bin_vals), color="black", alpha=1, zorder=100, linewidth=1)

        plt.legend()

        fig.savefig(os.path.join(path_to_data_dir, f"plots/n_patches/histogram/{grad_type}/{n_mocks}mocks/hist_n_patches_{nbins}bins_{dim[i]}.png"))
        plt.cla()

        print(f"histogram for n_patches, dim {dim[i]}, done")