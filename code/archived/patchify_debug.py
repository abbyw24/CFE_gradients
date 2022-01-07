from memory_profiler import profile
from mem_funcs import printsizeof

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import itertools as it
import os
from create_subdirs import create_subdirs
from ls_debug import xi_ls
from center_mock import center_mock
import generate_mock_list
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
lognormal_density = globals.lognormal_density
path_to_data_dir = globals.path_to_data_dir
grad_type = globals.grad_type

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_patches = globals.n_patches

mock_file_name_list = generate_mock_list.generate_mock_list()

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
@profile
def xi_in_patches(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, mock_file_name_list = mock_file_name_list, n_patches=n_patches, n=146):
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
        cond = (i >= n)
        if cond:
            print(f"mock {i}:")
        # retrieve mock info dictionary
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        printsizeof(mock_info, on="mock_info", cond=cond)
        mock_file_name = mock_info["mock_file_name"]
        mock_name = mock_info["mock_name"]
        mock_data = mock_info["grad_set"]
        L = mock_info["boxsize"]

        # center mock from 0 to L
        center_mock(mock_data, 0, L)

        nd = len(mock_data)

        # create random set
        nr = randmult*nd
        rand_set = np.random.uniform(0, L, (nr,3))
        printsizeof(rand_set, on="rand_set", cond=cond)

        # patchify mock data and random set
        patches_mock = patchify(mock_data, L, n_patches=n_patches)
        patch_ids_mock = patches_mock[0]
        patch_id_list_mock = np.unique(patch_ids_mock)

        patches_rand = patchify(rand_set, L, n_patches=n_patches)
        patch_ids_rand = patches_rand[0]
        patch_id_list_rand = np.unique(patch_ids_rand)
        if cond:
            print(f"patchify complete for mock {i}")

        # make sure patch lists match for mock and random, that there's nothing weird going on
        assert np.all(patch_id_list_mock == patch_id_list_rand)
        patch_id_list = patch_id_list_mock
        n_patches = len(patch_id_list)
        patches_idx = patches_mock[1]

        # define patch centers by taking mean of the random set in each patch
        patch_centers = []
        for patch_id in patch_id_list:
            patch_data = rand_set[patch_ids_rand == patch_id]
            center = np.mean(patch_data, axis=0)
            patch_centers.append(center)
        patch_centers = np.array(patch_centers)

        # whether to print statements from xi_ls
        prints = True if cond else False

        # results for full mock
        if cond:
            print(f"computing xi_ls for mock {i}...")
        results_xi_full = xi_ls(mock_data, rand_set, periodic, nthreads, rmin, rmax, nbins, prints=prints)
        printsizeof(results_xi_full, on="full xi results", cond=cond)
        xi_full = np.array(results_xi_full[1])
        if cond:
            print(f"full xi results computed for mock {i}")

        # save values used to calculate xi
        patch_info = {
            "periodic" : periodic,
            "nthreads" : nthreads,
            "rmin" : rmin,
            "rmax" : rmax,
            "nbins" : nbins
        }

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
            # # assert False right before script hangs
            # if cond and patch_id == 7:
            #     assert False
            # ###
            results_xi_patch = xi_ls(patch_data, patch_rand, periodic, nthreads, rmin, rmax, nbins, prints=prints)
            printsizeof(results_xi_patch, on=f"patch {patch_id} xi_ls", cond=cond)
            if cond:
                print(f"completed xi_ls for patch {patch_id}")
            xi_patch = results_xi_patch[1]

            plt.plot(r_avg, xi_patch, alpha=0.5, marker=".", label=patches_idx[k])
            xi_patches.append(xi_patch)
            k += 1
            if cond:
                print(f"patch {patch_id} xi results computed for mock {i}")
            printsizeof(cmap, on="cmap", cond=cond)
            printsizeof(fig, on="fig", cond=cond)
            printsizeof(ax, on="ax", cond=cond)
        xi_patches = np.array(xi_patches)
        printsizeof(xi_patches, on="xi_patches", cond=cond)

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
        if cond:
            print(f"saved patch info for mock {i}")

        # plot results
        plt.plot(r_avg, xi_full, color="black", marker=".", label="Full Mock")
        plt.plot(r_avg, xi_patch_avg, color="black", alpha=0.5, marker=".", label="Avg. of Patches")
        # plot parameters
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_box_aspect(1)
        ax.set_ylim((-0.01, 0.12))
        ax.set_xlabel(r'Separation $r$ ($h^{-1}\,$Mpc)')
        ax.set_ylabel(r'$\xi$(r)')
        plt.rcParams["axes.titlesize"] = 10
        if grad_type == "1mock":
            ax.set_title("")
        else:
            ax.set_title(f"Standard Estimator, Xi in Patches, {grad_dim}D, {mock_name}")
        plt.legend(prop={'size': 8})
        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches/{lognormal_density}/{n_patches}patches/xi/{mock_file_name}.png"))
        ax.cla()

        plt.close("all")

        print(f"xi in patches --> {mock_file_name}")

if __name__ == '__main__':
    xi_in_patches(n=21)