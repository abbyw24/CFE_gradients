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

def histogram_densities(densities_list, method, grad_type=grad_type, path_to_data_dir=path_to_data_dir, n_patches=n_patches, nbins=10):
    # make sure inputs have correct form
    assert method == "patches" or "suave"
    assert isinstance(densities_list, list)

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
    
    # get recovered and expected gradients
    grads_rec = {}
    grads_exp = {}
    all_grads = []       # to combine grads_rec from all patches; for bin edges

    for density in densities_list:
        grads_exp[str(density)] = []
        grads_rec[str(density)] = []
        for i in range(len(mock_file_name_list)):
            mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            grad_exp = mock_info["grad_expected"]
            grads_exp[str(density)].append(grad_exp)

            if method == "patches":
                info = np.load(os.path.join(path_to_data_dir, f"patch_data/{density}/{n_patches}patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
                grad_rec = info["grad_recovered"]
                grads_rec[str(density)].append(grad_rec)
            
            elif method == "suave":
                info = np.load(os.path.join(path_to_data_dir, f"suave_data/{density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
                grad_rec = info["grad_recovered"]
                grads_rec[str(density)].append(grad_rec)
            
            else:
                print("method must be either 'patches' or 'suave'")
                assert False

            all_grads.append(grad_rec-grad_exp)
    
    all_grads = np.array(all_grads)

    # loop through desired dimensions
    for i in dim:
        print(f"{dim[i]}:")
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {densities_list}, {dim[i]}, {grad_type}, {n_mocks} mocks")
        plt.xlabel("Recovered Grad. - Expected Grad.")
        plt.ylabel("Counts")

        # define bins
        bins = np.linspace(1.5*min(all_grads[:,i]), 1.5*max(all_grads[:,i]), nbins)

        a = 0.8
        bin_vals = []
        for density in densities_list:
            grads_rec_n = np.array(grads_rec[str(density)])
            grads_exp_n = np.array(grads_exp[str(density)])
            vals = grads_rec_n[:,i] - grads_exp_n[:,i]
            n, _, _ = plt.hist(vals, bins=bins, histtype="step", color="indigo", alpha=a, label=f"{density}")
            a /= 2.5
            bin_vals.append(n)
            print(f"for density {density}:")
            print("mean = ", np.mean(grads_rec_n[:,i]))
            print("min = ", np.min(grads_rec_n[:,i]))
            print("max = ", np.max(grads_rec_n[:,i]))
            print("std = ", np.std(grads_rec_n[:,i]))

        plt.legend()

        fig.savefig(os.path.join(path_to_data_dir, f"plots/densities/histogram/{grad_type}/{n_mocks}mocks/hist_densities_{nbins}bins_{dim[i]}.png"))
        plt.cla()

        print(f"histogram for densities, dim {dim[i]}, done")