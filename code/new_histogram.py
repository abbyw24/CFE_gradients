import numpy as np
import matplotlib.pyplot as plt
import os

from create_subdirs import create_subdirs
import new_globals

new_globals.initialize_vals()

grad_dim = new_globals.grad_dim
path_to_data_dir = new_globals.path_to_data_dir
mock_name_list = new_globals.mock_name_list

grad_type = new_globals.grad_type

n_patches = new_globals.n_patches

def histogram_patches_vs_suave(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches, label1="patches", label2="suave", nbins=10):
   # create the needed subdirectories
    sub_dirs = [
        "plots/patches_vs_suave/histogram"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)
    
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    grads_rec_patches = []
    grads_rec_suave = []

    # load in patches and suave info
    for i in range(len(mock_name_list)):
        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches/{n_patches}patches_{mock_name_list[i]}.npy"), allow_pickle=True).item()
        grad_rec_patches = patch_info["grad_recovered"]

        suave_info = np.load(os.path.join(path_to_data_dir, f"suave_data/{mock_name_list[i]}.npy"), allow_pickle=True).item()
        grad_rec_suave = suave_info["grad_recovered"]

        # append values to list of all mocks
        grads_rec_patches.append(grad_rec_patches)
        grads_rec_suave.append(grad_rec_suave)
    
    grads_rec_patches = np.array(grads_rec_patches)
    grads_rec_suave = np.array(grads_rec_suave)

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {dim[i]}, {grad_type}")
        plt.xlabel("Recovered Gradient")

        # line at x = 0
        plt.vlines(0, 0, 20, color="black", alpha=0.4)

        # histogram for patches
        plt.hist(grads_rec_patches[:,i], bins=nbins, alpha=0.5, label=label1)
        plt.hist(grads_rec_suave[:,i], bins=nbins, alpha=0.5, label=label2)

        plt.legend()

        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches_vs_suave/histogram/{grad_type}/hist_patches_vs_suave_{nbins}bins_{dim[i]}.png"))
        plt.cla()

        print(f"scatter plot for patches vs. suave, dim {dim[i]}, done")

histogram_patches_vs_suave()