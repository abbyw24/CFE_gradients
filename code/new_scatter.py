import numpy as np
import matplotlib.pyplot as plt
import os

import new_globals
from create_subdirs import create_subdirs

new_globals.initialize_vals()  # brings in all the default parameters

grad_dim = new_globals.grad_dim
path_to_data_dir = new_globals.path_to_data_dir
mock_name_list = new_globals.mock_name_list

n_patches = new_globals.n_patches

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

def scatter_patches_vs_suave(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches):
    # create the needed subdirectories
    sub_dirs = [
        "plots/patches_vs_suave/scatter"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }
    k = 0

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
    
    grads_exp = np.array(grads_exp)
    grads_rec_patches = np.array(grads_rec_patches)
    grads_rec_suave = np.array(grads_rec_suave)

    for i in dim:
        # create plot
        fig, ax = plt.subplots()
        ax.set_xlabel("Expected Gradient")
        ax.set_ylabel("Recovered Gradient")
        ax.set_title(f"Expected vs. Recovered Gradient, {dim[i]}")

        for j in range(len(grads_exp[:,i])):
            plt.plot(grads_exp[j,i], grads_rec_patches[j,i], marker=".", color="C0", alpha=0.5, label=label_p(k))
            plt.plot(grads_exp[j,i], grads_rec_suave[j,i], marker=".", color="orange", alpha=0.5, label=label_s(k))
            k += 1
        
        # plot line y = x (the data points would fall on this line if the expected and recovered gradients matched up perfectly)
        x = np.linspace(min(grads_exp[:,i]), max(grads_exp[:,i]), 10)
        plt.plot(x, x, color="black", alpha=0.5)
        plt.legend()
        
        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/scatter/scatter_patches_vs_suave_{dim[i]}.png"))
        plt.cla()
    
        print(f"scatter plot for patches vs. suave, dim {dim}, done")

scatter_patches_vs_suave()