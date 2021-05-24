import numpy as np
import matplotlib.pyplot as plt
import math
from create_subdirs import create_subdirs
import os
import new_globals

new_globals.initialize_vals()  # brings in all the default parameters

grad_dim = new_globals.grad_dim
path_to_data_dir = new_globals.path_to_data_dir
mock_name_list = new_globals.mock_name_list

nbins = new_globals.nbins
n_patches = new_globals.n_patches

# define function to find least square fit in every bin
def patches_lstsq_allbins(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches):
    # make sure all inputs have the right form
    assert isinstance(grad_dim, int)
    assert isinstance(path_to_data_dir, str)
    assert isinstance(n_patches, int)

    # create the needed subdirectories
    sub_dirs = [
        "plots/patches/lst_sq_fit/allbins"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    dim = ["x", "y", "z"]

    for i in range(len(mock_name_list)):
        # load in mock and patch info
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/dicts/{mock_name_list[i]}.npy"), allow_pickle=True).item()
        mock_name = mock_info["mock_name"]
        L = mock_info["boxsize"]
        m = mock_info["m"]
        b = mock_info["b"]

        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/patches_{mock_name_list[i]}.npy"), allow_pickle=True).item()
        patch_centers = patch_info["patch_centers"]
        patch_centers -= L/2
            # this centers the fiducial point in the box
        r_avg = patch_info["r_avg"]
        xi_patches = patch_info["xi_patches"]
        assert len(xi_patches) == n_patches
        xi_patch_avg = patch_info["xi_patch_avg"]
        xi_full = patch_info["xi_full"]

        # create A matrix
        A = np.ones(len(patch_centers))
        for i in range(len(dim)):
            A = np.c_[A, patch_centers[:,i]]

        # create C covariance matrix
            # for now, C = identity matrix
        C = np.identity(len(patch_centers))

        C_inv = np.linalg.inv(C)
    
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
    
    # add recovered values to patch info dictionary
    patch_info["b_fit"] = recovered_vals[0]
    patch_info["m_fit_x"] = recovered_vals[1]
    patch_info["m_fit_y"] = recovered_vals[2]
    patch_info["m_fit_z"] = recovered_vals[3]

    # resave patch info dictionary
    np.save(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches_{mock_name}"), patch_info, allow_pickle=True)

    plt.legend()
    fig.savefig(os.path.join(path_to_data_dir, f"plots/patches/lst_sq_fit/allbins/allbins_{n_patches}patches_{mock_name}.png"))
    ax.cla()

    print(f"least square fit in all bins, {mock_name}, {n_patches} patches")

patches_lstsq_allbins()