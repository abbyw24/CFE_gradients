import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import Corrfunc
import itertools as it
import os
from create_subdirs import create_subdirs
from corrfunc_ls import xi_ls

import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
lognormal_density = globals.lognormal_density
path_to_data_dir = globals.path_to_data_dir
mock_file_name_list = globals.mock_file_name_list
mock_name_list = globals.mock_name_list

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_patches = globals.n_patches

# define patchify
def patchify(data, boxsize, n_patches=n_patches):
    n_sides = n_patches**(1/3)
    assert n_sides.is_integer()
    n_sides = int(n_sides)
    nd, n_dim = data.shape
    boxsize_patch = boxsize/n_sides
    a = np.arange(n_sides)
    idx_patches = np.array(list(it.product(a, repeat=n_dim)))
    patch_ids = np.zeros(nd).astype(int)
    for patch_id,iii in enumerate(idx_patches):
        # define the boundaries of the patch
        mins = iii*boxsize_patch
        maxes = (iii+1)*boxsize_patch
        # define mask as where all of the values must be within the boundaries
        mask_min = np.array([(data[:,d]<maxes[d]) for d in range(n_dim)])
        mask_max = np.array([(data[:,d]>=mins[d]) for d in range(n_dim)])
        mask = np.vstack([mask_min, mask_max]).T
        mask_combined = np.all(mask, axis=1)
        # perform masking and save to array
        patch_ids[mask_combined] = patch_id
    return patch_ids, idx_patches

# define function to find xi in each patch
def xi_in_patches(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, mock_file_name_list = mock_file_name_list, n_patches=n_patches):
    # make sure all inputs have the right form
    assert isinstance(grad_dim, int)
    assert isinstance(path_to_data_dir, str)
    assert isinstance(n_patches, int)

    # create the needed subdirectories
    sub_dirs = [
        f"patch_data/{lognormal_density}/{n_patches}patches",
        f"plots/patches/{lognormal_density}/{n_patches}patches/xi"
    ]
    create_subdirs(f"{path_to_data_dir}", sub_dirs)

    for i in range(len(mock_file_name_list)):
        # retrieve mock info dictionary
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        mock_file_name = mock_info["mock_file_name"]
        mock_name = mock_info["mock_name"]
        mock_data = mock_info["grad_set"]
        L = mock_info["boxsize"]

        # if there are negative values, shift by L/2, to 0 to L
        if np.any(mock_data <= 0):
            mock_data += L/2
        else:
            print("input mock data must be from -L/2 to L/2 (values shifted during xi_in_patches)")
            assert False

        nd = len(mock_data)

        # create random set
        nr = randmult*nd
        rand_set = np.random.uniform(0, L, (nr,3))

        # patchify mock data and random set
        patches_mock = patchify(mock_data, L, n_patches=n_patches)
        patch_ids_mock = patches_mock[0]
        patch_id_list_mock = np.unique(patch_ids_mock)

        patches_rand = patchify(rand_set, L, n_patches=n_patches)
        patch_ids_rand = patches_rand[0]
        patch_id_list_rand = np.unique(patch_ids_rand)

        # make sure patch lists match for mock and random, that there's nothing weird going on
        assert np.all(patch_id_list_mock == patch_id_list_rand)
        patch_id_list = patch_id_list_mock
        n_patches = len(patch_id_list)
        patches_idx = patches_mock[1]

        # create a dictionary for patch data
        patch_info = {}
        # define patch centers by taking mean of the random set in each patch
        patch_centers = []
        for patch_id in patch_id_list:
            patch_data = rand_set[patch_ids_rand == patch_id]
            center = np.mean(patch_data, axis=0)
            patch_centers.append(center)
        patch_centers = np.array(patch_centers)

        # results for full mock
        results_xi_full = xi_ls(mock_data, rand_set, periodic, nthreads, rmin, rmax, nbins)
        xi_full = np.array(results_xi_full[1])
        # save values used to calculate xi
        patch_data["nthreads"] = nthreads
        patch_data["rmin"] = rmin
        patch_data["rmax"] = rmax
        patch_data["nbins"] = nbins

        # define r_avg (this is the same for all xi)
        r_avg = results_xi_full[0]

        fig, ax = plt.subplots()

        # results in patches
        xi_patches = []
        k = 0
        cmap = plt.cm.get_cmap("cool")
        ax.set_prop_cycle('color', cmap(np.linspace(0, 1, n_patches)))

        for patch_id in patch_id_list:
            patch_data = mock_data[patch_ids_mock == patch_id]
            patch_rand = rand_set[patch_ids_rand == patch_id]
            results_xi_patch = xi_ls(patch_data, patch_rand, periodic, nthreads, rmin, rmax, nbins)
            xi_patch = results_xi_patch[1]

            plt.plot(r_avg, xi_patch, alpha=0.5, marker=".", label=patches_idx[k])
            xi_patches.append(xi_patch)
            k += 1
        xi_patches = np.array(xi_patches)

        # average of patch results
        xi_patch_avg = np.sum(xi_patches, axis=0)/len(xi_patches)

        # save xi data– to load in separate file for least square fit
        patch_info = {
            "n_patches" : n_patches,
            "patch_centers" : patch_centers,
            "r_avg" : r_avg,
            "xi_patches" : xi_patches,
            "xi_patch_avg" : xi_patch_avg,
            "xi_full" : xi_full
            }
        np.save(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name}"), patch_info, allow_pickle=True)

        # plot results
        plt.plot(r_avg, xi_full, color="black", marker=".", label="Full Mock")
        plt.plot(r_avg, xi_patch_avg, color="black", alpha=0.5, marker=".", label="Avg. of Patches")
        # plot parameters
        ax.set_xlabel(r'r ($h^{-1}$Mpc)')
        ax.set_ylabel(r'$\xi$(r)')
        plt.rcParams["axes.titlesize"] = 10
        ax.set_title(f"Standard Estimator, Xi in Patches, {grad_dim}D, {mock_name}")
        plt.legend(prop={'size': 8})
        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches/{lognormal_density}/{n_patches}patches/xi/{mock_file_name}.png"))
        ax.cla()

        plt.close("all")

        print(f"xi in patches --> {mock_file_name}")