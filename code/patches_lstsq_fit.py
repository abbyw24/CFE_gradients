import numpy as np
import matplotlib.pyplot as plt
import math
from create_subdirs import create_subdirs
import os
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
path_to_data_dir = globals.path_to_data_dir
mock_name_list = globals.mock_name_list

nbins = globals.nbins
n_patches = globals.n_patches

# define function to find least square fit in every bin
def patches_lstsq_allbins(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches, bin_cutoff_val=2):
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
        grad_expected = mock_info["grad_expected"]

        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches/{n_patches}patches_{mock_name_list[i]}.npy"), allow_pickle=True).item()
        patch_centers = patch_info["patch_centers"]
        patch_centers -= L/2
            # this centers the fiducial point in the box
        r_avg = patch_info["r_avg"]
        xi_patches = patch_info["xi_patches"]
        assert len(xi_patches) == n_patches

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
        # figure this out for grad_dim > 1
        ax.axhline(grad_expected[0], color="red", alpha=0.5)

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
        bin_cutoff = int(nbins/bin_cutoff_val)
        # plot bin cutoff
        ax.vlines(r_avg[bin_cutoff], -0.05, 0.05, alpha=0.2, linestyle="dashed", label="Cutoff for grad calculation")

        recovered_vals = []
        for value in fit_vals:
            fits_rec = value[:bin_cutoff]
            val_rec = np.mean(fits_rec)
                # average of the fit value in each bin up to cutoff
            recovered_vals.append(val_rec)
        b_fit = recovered_vals[0]
        m_fit = recovered_vals[1:]
    
        # add recovered values to patch info dictionary
        patch_info["b_fit"] = b_fit
        patch_info["m_fit"] = m_fit

        # add recovered gradient value to patch info dictionary
        grad_recovered = m_fit/b_fit
        patch_info[f"grad_recovered"] = grad_recovered

        # ratio of recovered to expected
        ratio_rec_exp = grad_recovered[0]/grad_expected[0]
        # again, need to figure this out for grad_dim > 1 !
        patch_info["ratio_rec_exp"] = ratio_rec_exp

        # resave patch info dictionary
        np.save(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches/{n_patches}patches_{mock_name}"), patch_info, allow_pickle=True)

        plt.legend()
        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches/lst_sq_fit/allbins/allbins_{n_patches}patches_{mock_name}.png"))
        ax.cla()

        plt.close("all")

        print(f"least square fit in all bins, {mock_name}, {n_patches} patches")

def patches_lstsq_fit_1bin(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches, r_bin=2):
    # make sure all inputs have the right form
    assert isinstance(path_to_data_dir, str)
    for i in [grad_dim, n_patches, r_bin]:
        assert isinstance(i, int)

    # create the needed subdirectories
    sub_dirs = [
        f"plots/patches/lst_sq_fit/bin{r_bin}"
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
        grad_expected = mock_info["grad_expected"]

        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches/{n_patches}patches_{mock_name_list[i]}.npy"), allow_pickle=True).item()
        patch_centers = patch_info["patch_centers"]
        patch_centers -= L/2
            # this centers the fiducial point in the box
        r_avg = patch_info["r_avg"]
        xi_patches = patch_info["xi_patches"]
        assert len(xi_patches) == n_patches

        # create A matrix
        A = np.ones(len(patch_centers))
        for i in range(len(dim)):
            A = np.c_[A, patch_centers[:,i]]

        # create C covariance matrix
            # for now, C = identity matrix
        C = np.identity(len(patch_centers))

        C_inv = np.linalg.inv(C)

        # clustering amplitudes
        Y = xi_patches[:,r_bin-1]
        r_avg_bin = r_avg[r_bin-1]*np.ones(n_patches)

        # calculate matrix X = [b,m]
        fig2, ax2 = plt.subplots()
        # color map– color code points to match corresponding patch center in grad_xi figure
        cmap = plt.cm.get_cmap("cool")
        ax2.set_prop_cycle('color',cmap(np.linspace(0,1,n_patches)))
        plt.scatter(patch_centers[:,0], Y, marker="o", c=Y, cmap="cool", label=f"Mock: {grad_dim}D, {mock_name}")
        ax2.set_xlabel(r"Patch Centers ($h^{-1}$Mpc)")
        ax2.set_ylabel(r"$\xi$(r)")
        x = np.linspace(min(patch_centers[:,0]),max(patch_centers[:,0]))

        # set colors for best fit lines
        bestfit_colors = ["blue", "grey", "silver"]

        # perform least square fit in specified r_bin
        X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)
        b_fit = X[0]
        m_fit = X[1:]
    
        # add recovered values to patch info dictionary
        patch_info[f"b_fit_bin{r_bin}"] = b_fit
        patch_info[f"m_fit_bin{r_bin}"] = m_fit
        
        # add recovered gradient value to patch info dictionary
        grad_recovered = m_fit/b_fit
        patch_info[f"grad_recovered_bin{r_bin}"] = grad_recovered

        # ratio of recovered to expected
        ratio_rec_exp = grad_recovered[0]/grad_expected[0]
        # again, need to figure this out for grad_dim > 1 !
        patch_info[f"ratio_rec_exp_bin{r_bin}"] = ratio_rec_exp

        # plot results
        for i in range(len(dim)):
            plt.plot(x, X[i+1]*x + b_fit, color=bestfit_colors[i], label=dim[i]+" best fit: y = "+str("%.8f" %X[i+1])+"x + "+str("%.6f" %b_fit))
        plt.plot(patch_centers[0,0], Y[0], alpha=0.0, label="{:.8f}".format(m/(b*L)))
        ax2.set_title(f"Linear least square fit, Clustering amps in patches (bin {r_bin}); \n Expected: {grad_expected[0]}, Recovered: {grad_recovered[0]}, Ratio: {ratio_rec_exp}")
        plt.legend()

        # resave patch info dictionary
        np.save(os.path.join(path_to_data_dir, f"patch_data/{n_patches}patches/{n_patches}patches_{mock_name}"), patch_info, allow_pickle=True)

        # save figure
        fig2.savefig(os.path.join(path_to_data_dir, f"plots/patches/lst_sq_fit/bin{r_bin}/bin{r_bin}_{n_patches}patches_{mock_name}.png"))
        ax2.cla()
        plt.close("all")

        print(f"least square fit in bin {r_bin}, {mock_name}, {n_patches} patches")

patches_lstsq_allbins()

patches_lstsq_fit_1bin(r_bin=2)