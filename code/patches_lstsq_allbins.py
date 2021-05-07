import numpy as np
import matplotlib.pyplot as plt
import math
from create_subdirs import create_subdirs
import os
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
L = globals.L
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

nbins = globals.nbins

n_patches = globals.n_patches

# define function to find least square fit in every bin
def patches_lstsq_allbins(grad_dim, m, b, path_to_mocks_dir, mock_name, n_patches=n_patches, nbins=nbins):
    # make sure all inputs have the right form
    assert isinstance(path_to_mocks_dir, str)
    assert isinstance(mock_name, str)
    for x in [grad_dim, m, b]:
        assert isinstance(x, (int, float))

    # create the needed subdirectories
    create_subdirs(f"{path_to_mocks_dir}/patches", ["patch_centers", "xi", "plots", "lst_sq_fit"])

    dim = ["x", "y", "z"]
    patch_centers = np.load(os.path.join(path_to_mocks_dir, f"patches/patch_centers/patch_centers_{mock_name}.npy"))
    patch_centers -= L/2
        # this centers the fiducial point in the box

    # create A matrix
    A = np.ones(len(patch_centers))
    for i in range(len(dim)):
        A = np.c_[A, patch_centers[:,i]]

    # create C covariance matrix
        # for now, C = identity matrix
    C = np.identity(len(patch_centers))

    C_inv = np.linalg.inv(C)

    # Y matrix = clustering amplitudes
    patchify_data = np.load(os.path.join(path_to_mocks_dir, f"patches/xi/xi_{n_patches}patches_{mock_name}.npy"), allow_pickle=True).item()
    r_avg = patchify_data["r_avg"]
    xi_patches = patchify_data["xi_patches"]
    xi_patch_avg = patchify_data["xi_patch_avg"]
    xi_full = patchify_data["xi_full"]

    assert len(xi_patches) == n_patches

    # plot xi_patches
    fig, ax = plt.subplots()
    # plot xi in each patch across all bins
    cmap = plt.cm.get_cmap("cool")
    ax.set_prop_cycle('color', cmap(np.linspace(0, 1, n_patches)))
    for patch in xi_patches:
        plt.plot(r_avg, patch, alpha=0.5, marker=".")

    ax.set_title(f"Clustering amps in patches, {mock_name}")
    ax.set_xlabel(r"r ($h^{-1}$Mpc)")
    ax.set_ylabel(r"$\xi$(r)")

    # expected "strength of gradient"
    grad_expected = m/(b*L)
    ax.axhline(grad_expected, color="red", alpha=0.5)

    fits = []
    for r_bin in range(nbins):
        # clustering amplitudes
        Y = xi_patches[:,r_bin]
        # least square fit
        X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)
            # X gives best fit: [b, m_x, m_y, m_z]
        fits.append(X)
    fit_vals = np.array(fits).T

    # plot m_fit/b_fit in each bin
    #       m_fit_x/b_fit should match grad_expected, and y and z should be zero
    plt.plot(r_avg, fit_vals[1]/fit_vals[0], color="black", marker=".", label="x fit")
    plt.plot(r_avg, fit_vals[2]/fit_vals[0], color="black", marker=".", alpha=0.6, label="y fit")
    plt.plot(r_avg, fit_vals[3]/fit_vals[0], color="black", marker=".", alpha=0.4, label="z fit")

    # create our recovered gradient array (as of now with a set n_bin cutoff to avoid too much noise)
    bin_cutoff = int(nbins/2)
    # plot bin cutoff
    ax.vlines(r_avg[bin_cutoff], -0.05, 0.05, alpha=0.2, linestyle="dashed", label="Cutoff for grad calculation")

    recovered_vals = []
    for value in fit_vals:
        fits_rec = value[:bin_cutoff]
        val_rec = np.mean(fits_rec)
            # average of the fit value in each bin up to cutoff
        recovered_vals.append(val_rec)
    
    recovered_values = {
        "b_fit" : recovered_vals[0],
        "m_fit_x" : recovered_vals[1],
        "m_fit_y" : recovered_vals[2],
        "m_fit_z" : recovered_vals[3]
    }

    # save recovered gradient values
    np.save(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/recovered_vals_{n_patches}patches_{mock_name}"), recovered_values)
    print(f"least square fit in all bins, {mock_name}, {n_patches} patches")

    plt.legend()
    fig.savefig(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/allbins_{n_patches}patches_{mock_name}.png")
    plt.cla()