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