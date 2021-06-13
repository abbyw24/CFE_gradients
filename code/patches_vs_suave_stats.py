import numpy as np
import matplotlib.pyplot as plt
import os

import globals
from create_subdirs import create_subdirs

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
path_to_data_dir = globals.path_to_data_dir
mock_file_name_list = globals.mock_file_name_list
lognormal_density = globals.lognormal_density
n_mocks = globals.n_mocks

grad_type = globals.grad_type

n_patches = globals.n_patches

def scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, grad_type=grad_type, path_to_data_dir=path_to_data_dir,
    lognormal_density = lognormal_density, n_patches=n_patches):

    # create the needed subdirectories
    sub_dirs = [
        f"plots/patches_vs_suave/scatter/{lognormal_density}/{grad_type}/{n_mocks}mocks"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    for i in dim:
        # create plot
        fig, ax = plt.subplots()
        ax.set_xlabel("Expected Gradient")
        ax.set_ylabel("Recovered Gradient")
        ax.set_title(f"Expected vs. Recovered Gradient, {dim[i]}, {grad_type}, {lognormal_density}, {n_mocks} mocks")

        plt.scatter(grads_exp[:,i], grads_rec_suave[:,i], marker=".", color="indigo", alpha=0.6, label="CFE")
        plt.scatter(grads_exp[:,i], grads_rec_patches[:,i], marker=".", color="gray", alpha=0.6, label="Standard")
        
        # plot line y = x (the data points would fall on this line if the expected and recovered gradients matched up perfectly)
        x = np.linspace(min(grads_exp[:,i]), max(grads_exp[:,i]), 10)
        plt.plot(x, x, color="black", alpha=0.5)
        plt.legend()
        
        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/scatter/{lognormal_density}/{grad_type}/{n_mocks}mocks/scatter_{n_patches}patches_vs_suave_{dim[i]}.png"))
        plt.cla()
    
        print(f"scatter plot for patches vs. suave, dim {dim[i]}, done")

def histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, grad_type=grad_type, path_to_data_dir=path_to_data_dir,
    lognormal_density=lognormal_density, n_patches=n_patches, nbins=30):

    # create the needed subdirectories
    sub_dirs = [
        f"plots/patches_vs_suave/histogram/{lognormal_density}/{grad_type}/{n_mocks}mocks"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)
    
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig, ax = plt.subplots()
        if grad_type == "1mock":
            ax.set_title("")
        else:
            ax.set_title(f"Histogram of Recovered Gradient, {dim[i]}, {grad_type}, {lognormal_density}, {n_mocks} mocks")
        ax.set_xlabel("Recovered Gradient - True Gradient ($h\,$Mpc$^{-1}$)")
        ax.set_ylabel("Counts")

        suave_vals = grads_rec_suave[:,i]-grads_exp[:,i]
        patches_vals = grads_rec_patches[:,i]-grads_exp[:,i]

        # define bins
        bins = np.linspace(min(min(patches_vals), min(suave_vals)), max(max(patches_vals), max(suave_vals)), nbins)
        n_s, _, _ = plt.hist(suave_vals, bins=bins, color="forestgreen", alpha=0.8, label="CFE")
        n_p, _, _ = plt.hist(patches_vals, bins=bins, color="black", alpha=0.2, label="Standard", zorder=100)

        plt.vlines(0, 0, max(max(n_s), max(n_p)), color="black", alpha=1, zorder=101, linewidth=1)

        plt.legend()

        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/histogram/{lognormal_density}/{grad_type}/{n_mocks}mocks/hist_{n_patches}patches_vs_suave_{nbins}bins_{dim[i]}.png"))
        plt.cla()

        print(f"histogram for patches vs. suave, dim {dim[i]}, done")

def extract_grads_patches_suave():
    grads_exp = []
    grads_rec_patches = []
    grads_rec_suave = []

    # load in mock, patches, and suave info
    for i in range(len(mock_file_name_list)):
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        mock_file_name = mock_info["mock_file_name"]
        grad_expected = mock_info["grad_expected"]
        
        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        grad_rec_patches = patch_info["grad_recovered"]

        suave_info = np.load(os.path.join(path_to_data_dir, f"suave_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        grad_rec_suave = suave_info["grad_recovered"]

        # append values to list of all mocks
        grads_exp.append(grad_expected)
        grads_rec_patches.append(grad_rec_patches)
        grads_rec_suave.append(grad_rec_suave)

    grads = {
        "grads_exp" : np.array(grads_exp),
        "grads_rec_patches" : np.array(grads_rec_patches),
        "grads_rec_suave" : np.array(grads_rec_suave)
    }

    return grads

def stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave, grad_type=grad_type, path_to_data_dir=path_to_data_dir,
    lognormal_density=lognormal_density):

    n_mocks = len(grads_exp)
    # create the needed subdirectories
    sub_dirs = [
        f"patches_vs_suave_data/{lognormal_density}/{grad_type}/{n_mocks}mocks"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    print(f"for grad type {grad_type}, {len(grads_exp)} mocks:")
    for i in dim:
        print(f"{dim[i]}:")
        # mean
        mean_patches = np.mean(grads_rec_patches[:,i])
        print(f"mean rec. grad., patches = {mean_patches}")
        mean_suave = np.mean(grads_rec_suave[:,i])
        print(f"mean rec. grad., suave = {mean_suave}")
        # min
        min_patches = min(grads_rec_patches[:,i])
        print(f"min rec. grad., patches = {min_patches}")
        min_suave = min(grads_rec_suave[:,i])
        print(f"min rec. grad., suave = {min_suave}")
        # max
        max_patches = max(grads_rec_patches[:,i])
        print(f"max rec. grad., patches = {max_patches}")
        max_suave = max(grads_rec_suave[:,i])
        print(f"max rec. grad., suave = {max_suave}")
        # median
        median_patches = np.median(grads_rec_patches[:,i])
        print(f"median rec. grad., patches = {median_patches}")
        median_suave = np.median(grads_rec_suave[:,i])
        print(f"median rec. grad., suave = {median_suave}")
        # standard deviation
        std_patches = np.std(grads_rec_patches[:,i])
        print(f"std rec. grad., patches = {std_patches}")
        std_suave = np.std(grads_rec_suave[:,i])
        print(f"std rec. grad., suave = {std_suave}")

        # save to dictionary
        stats = {
            "mean_patches" : mean_patches,
            "mean_suave" : mean_suave,
            "median_patches" : median_patches,
            "median_suave" : median_suave,
            "std_patches" : std_patches,
            "std_suave" : std_suave,
        }

        np.save(os.path.join(path_to_data_dir, f"patches_vs_suave_data/{lognormal_density}/{grad_type}/{n_mocks}mocks/stats_{dim[i]}"), stats)