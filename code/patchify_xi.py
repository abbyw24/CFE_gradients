import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import Corrfunc
import itertools as it
import os
from create_subdirs import create_subdirs

import globals

globals.initialize_vals()  # brings in all the default parameters

periodic = globals.periodic
randmult = globals.randmult
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

# define Corrfunc Landy-Szalay
def xi(data, rand_set, periodic=False, rmin=20.0, rmax=100.0, nbins=22, nthreads=12):
    # parameters
    r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    nr = len(rand_set)

    x, y, z = data[:,0], data[:,1], data[:,2]
    x_rand, y_rand, z_rand = rand_set[:,0], rand_set[:,1], rand_set[:,2]

    dd_res = Corrfunc.theory.DD(1, nthreads, r_edges, x, y, z, periodic=periodic)
    dr_res = Corrfunc.theory.DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, periodic=periodic)
    rr_res = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=periodic)

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd,nd,nr,nr,dd,dr,dr,rr)

    return r_avg, results_xi

# define function to find xi in each patch
def xi_in_patches(grad_dim, path_to_mocks_dir, mock_name, n_patches=n_patches, randmult=randmult):
    # make sure all inputs have the right form
    assert isinstance(path_to_mocks_dir, str)
    assert isinstance(mock_name, str)

    # create the needed subdirectories
    create_subdirs(f"{path_to_mocks_dir}/patches", ["patch_centers", "xi", "plots/xi"])

    # check that there is a corresponding boxsize file
    boxsize_file = os.path.join(path_to_mocks_dir, f"boxsize.npy")
    assert os.path.exists(boxsize_file)

    # load in mock data and boxsize
    mock_data = np.load(os.path.join(path_to_mocks_dir, f"gradmock_data/gradmock_data_{mock_name}.npy"))
    L = np.load(boxsize_file)

    # if there are negative values, shift by L/2, to 0 to L
    if np.any(mock_data <= 0):
        mock_data += L/2
    else:
        print("input mock data must be from -L/2 to L/2 (values shifted during xi_in_patches")
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

    # define patch centers by taking mean of the random set in each patch
    patch_centers = []
    for patch_id in patch_id_list:
        patch_data = rand_set[patch_ids_rand == patch_id]
        center = np.mean(patch_data, axis=0)
        patch_centers.append(center)
    patch_centers = np.array(patch_centers)
    np.save(os.path.join(path_to_mocks_dir, f"patches/patch_centers/patch_centers_{mock_name}"), patch_centers)

    # results for full mock
    results_xi_full = xi(mock_data, rand_set)
    xi_full = np.array(results_xi_full[1])

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
        results_xi_patch = xi(patch_data, patch_rand)
        xi_patch = results_xi_patch[1]

        plt.plot(r_avg, xi_patch, alpha=0.5, marker=".", label=patches_idx[k])
        xi_patches.append(xi_patch)
        k += 1
    xi_patches = np.array(xi_patches)

    # average of patch results
    xi_patch_avg = np.sum(xi_patches, axis=0)/len(xi_patches)

    # save xi data– to load in separate file for least square fit
    patches_xi = {
        "r_avg" : r_avg,
        "xi_patches" : xi_patches,
        "xi_patch_avg" : xi_patch_avg,
        "xi_full" : xi_full
        }
    np.save(os.path.join(path_to_mocks_dir, f"patches/xi/xi_{n_patches}patches_{mock_name}"), patches_xi, allow_pickle=True)

    # plot results
    plt.plot(r_avg, xi_full, color="black", marker=".", label="Full Mock")
    plt.plot(r_avg, xi_patch_avg, color="black", alpha=0.5, marker=".", label="Avg. of Patches")
    # plot parameters
    ax.set_xlabel(r'r ($h^{-1}$Mpc)')
    ax.set_ylabel(r'$\xi$(r)')
    plt.rcParams["axes.titlesize"] = 10
    ax.set_title(f"Standard Estimator, Xi in Patches, {grad_dim}D, {mock_name}")
    plt.legend(prop={'size': 8})
    fig.savefig(os.path.join(path_to_mocks_dir, f"patches/plots/xi/xi_{n_patches}patches_{mock_name}.png"))
    ax.cla()

    plt.close("all")