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

def patches_lstsq_fit(grad_dim, m, b, path_to_mocks_dir, mock_name, n_patches=n_patches, r_bin=2, nbins=nbins):
    # make sure all inputs have the right form
    assert isinstance(path_to_mocks_dir, str)
    assert isinstance(mock_name, str)
    for x in [grad_dim, m, b]:
        assert isinstance(x, (int, float))

    # create the needed subdirectories
    create_subdirs(f"{path_to_mocks_dir}/patches", ["patch_centers", "xi", f"plots/bin{r_bin}", "lst_sq_fit"])

    dim = ["x", "y", "z"]
    patch_centers = np.load(os.path.join(path_to_mocks_dir, f"patches/patch_centers/patch_centers_{mock_name}.npy"))
    patch_centers -= L/2

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

    # clustering amplitudes
    Y = xi_patches[:,r_bin-1]
    r_avg_bin = r_avg[r_bin-1]*np.ones(n_patches)

    # # plot xi_patches
    # fig1, ax1 = plt.subplots()
    # ax1.set_title(f"Clustering amps in patches, {mock_name}")
    # ax1.set_xlabel(r"r ($h^{-1}$Mpc)")
    # ax1.set_ylabel(r"$\xi$(r)")
    # cmap = plt.cm.get_cmap("cool")
    # ax1.set_prop_cycle('color',cmap(np.linspace(0,1,n_patches)))
    # for patch in xi_patches:
    #     plt.plot(r_avg, patch, alpha=0.5, marker=".")
    # plt.scatter(r_avg_bin, Y, alpha=0.5, color="black")
    # plt.legend()
    # ax1.cla()

    # calculate matrix X = [b,m]
    fig2, ax2 = plt.subplots()
    # color map– color code points to match corresponding patch center in grad_xi figure
    cmap = plt.cm.get_cmap("cool")
    ax2.set_prop_cycle('color',cmap(np.linspace(0,1,n_patches)))
    plt.scatter(patch_centers[:,0], Y, marker="o", c=Y, cmap="cool", label=f"Mock: {grad_dim}D, {mock_name}")
    ax2.set_title(f"Linear least square fit, Clustering amps in patches (bin {r_bin})")
    ax2.set_xlabel(r"Patch Centers ($h^{-1}$Mpc)")
    ax2.set_ylabel(r"$\xi$(r)")
    x = np.linspace(min(patch_centers[:,0]),max(patch_centers[:,0]))

    # set colors for best fit lines
    bestfit_colors = ["blue", "grey", "silver"]
    X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)
    b_fit = X[0]
    for i in range(len(dim)):
        plt.plot(x, X[i+1]*x + b_fit, color=bestfit_colors[i], label=dim[i]+" best fit: y = "+str("%.8f" %X[i+1])+"x + "+str("%.6f" %b_fit))
    plt.plot(patch_centers[0,0], Y[0], alpha=0.0, label="{:.8f)}".format(m/(b*L)))
    plt.legend()

    # save figure
    fig2.savefig(os.path.join(path_to_mocks_dir, f"patches/plots/bin-{r_bin}_{n_patches}patches_{mock_name}.png"))
    ax2.cla()
    plt.close("all")

    print(f"least square fit in bin {r_bin}, {mock_name}, {n_patches} patches")