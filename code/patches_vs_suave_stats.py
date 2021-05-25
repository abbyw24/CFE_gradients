import numpy as np
import matplotlib.pyplot as plt
import os

import globals
from create_subdirs import create_subdirs

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
path_to_data_dir = globals.path_to_data_dir
mock_name_list = globals.mock_name_list

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

def scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, grad_type=grad_type, path_to_data_dir=path_to_data_dir):
    # create the needed subdirectories
    sub_dirs = [
        f"plots/patches_vs_suave/scatter/{grad_type}"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }
    k = 0

    for i in dim:
        # create plot
        fig, ax = plt.subplots()
        ax.set_xlabel("Expected Gradient")
        ax.set_ylabel("Recovered Gradient")
        ax.set_title(f"Expected vs. Recovered Gradient, {dim[i]}, {grad_type}")

        for j in range(len(grads_exp[:,i])):
            plt.plot(grads_exp[j,i], grads_rec_patches[j,i], marker=".", color="C0", alpha=0.5, label=label_p(k))
            plt.plot(grads_exp[j,i], grads_rec_suave[j,i], marker=".", color="orange", alpha=0.5, label=label_s(k))
            k += 1
        
        # plot line y = x (the data points would fall on this line if the expected and recovered gradients matched up perfectly)
        x = np.linspace(min(grads_exp[:,i]), max(grads_exp[:,i]), 10)
        plt.plot(x, x, color="black", alpha=0.5)
        plt.legend()
        
        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/scatter/{grad_type}/scatter_patches_vs_suave_{dim[i]}.png"))
        plt.cla()
    
        print(f"scatter plot for patches vs. suave, dim {dim[i]}, done")

def histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, grad_type=grad_type, path_to_data_dir=path_to_data_dir, label1="patches", label2="suave", nbins=10):
   # create the needed subdirectories
    sub_dirs = [
        f"plots/patches_vs_suave/histogram/{grad_type}"
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
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {dim[i]}, {grad_type}")
        plt.xlabel("Recovered Gradient")

        # line at x = 0
        plt.vlines(0, 0, 20, color="black", alpha=0.4)

        # define bins
        bins = np.linspace(1.5*min(grads_rec_patches[:,i]), 1.5*max(grads_rec_patches[:,i]), nbins)
        plt.hist(grads_rec_patches[:,i], bins=bins, alpha=0.5, label=label1)
        plt.hist(grads_rec_suave[:,i], bins=bins, alpha=0.5, label=label2)

        plt.legend()

        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/histogram/{grad_type}/hist_patches_vs_suave_{nbins}bins_{dim[i]}.png"))
        plt.cla()

        print(f"scatter plot for patches vs. suave, dim {dim[i]}, done")

def extract_grads_patches_suave():
    grads_exp = []
    grads_rec_patches = []
    grads_rec_suave = []

    # load in mock, patches, and suave info
    for i in range(len(mock_name_list)):
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/dicts/{mock_name_list[i]}.npy"), allow_pickle=True).item()
        mock_name = mock_info["mock_name"]
        grad_expected = mock_info["grad_expected"]
        
        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches/{n_patches}patches_{mock_name_list[i]}.npy"), allow_pickle=True).item()
        grad_rec_patches = patch_info["grad_recovered"]

        suave_info = np.load(os.path.join(path_to_data_dir, f"suave_data/{mock_name_list[i]}.npy"), allow_pickle=True).item()
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

def stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave, grad_type=grad_type, path_to_data_dir=path_to_data_dir):
    n_mocks = len(grads_exp)
    # create the needed subdirectories
    sub_dirs = [
        f"patches_vs_suave_data/{grad_type}/{n_mocks}mocks"
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

        np.save(os.path.join(path_to_data_dir, f"patches_vs_suave_data/{grad_type}/{n_mocks}mocks/stats_{dim[i]}"), stats)


grads = extract_grads_patches_suave()
grads_exp = grads["grads_exp"]
grads_rec_patches = grads["grads_rec_patches"]
grads_rec_suave = grads["grads_rec_suave"]

scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave)

histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, nbins=20)

stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave)