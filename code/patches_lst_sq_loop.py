import numpy as np
import matplotlib.pyplot as plt
import math
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
L = globals.L
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_sides = globals.n_sides
n_patches = n_sides**3
    # since this script uses absolute number of patches instead of number of patches per side length:


# loop through m and b values
for m in m_arr_perL:
    for b in b_arr:
        dim = ["x", "y", "z"]
        patch_centers = np.load(f"gradient_mocks/{grad_dim}D/patches/patch_centers/patch_centers_m-{m}-L_b-{b}_{n_patches}patches.npy")
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
        patchify_data = np.load(f"gradient_mocks/{grad_dim}D/patches/grad_xi_m-{m}-L_b-{b}_{n_patches}patches.npy", allow_pickle=True)
        r_avg = patchify_data["r_avg"]
        xi_patches = patchify_data["xi_patches"]
        xi_patch_avg = patchify_data["xi_patch_avg"]
        xi_full = patchify_data["xi_full"]

        assert len(xi_patches) == n_patches

        # plot xi_patches
        fig1 = plt.figure()
        plt.title(f"Clustering amps in patches, m={m}, b={b}")
        plt.xlabel(r"r ($h^{-1}$Mpc)")
        plt.ylabel(r"$\xi$(r)")

        # expected "strength of gradient"!
        grad_expected = m/(b*L)
        plt.axhline(grad_expected)
        cmap = plt.cm.get_cmap("cool")
        ax = plt.axes()
        ax.set_prop_cycle('color',cmap(np.linspace(0,1,n_patches)))
        for patch in xi_patches:
            plt.plot(r_avg, patch, alpha=0.5, marker=".")

        # m_fits_x = []
        # m_fits_y = []
        # m_fits_z = []
        # b_fits = []
        fits = []
        for r_bin in range(nbins):
            # clustering amplitudes
            Y = xi_patches[:,r_bin]
            # least square fit
            X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)
            fits.append(X)
            # m_fits_x.append(X[1])
            # m_fits_y.append(X[2])
            # m_fits_z.append(X[3])
            # b_fits.append(X[0])
        print(fits)
        print(fits[0])
        assert False
        #fit_vals = [m_fits_x, m_fits_y, m_fits_z, b_fits]

        # create our recovered gradient array (as of now with a set n_bin cutoff to avoid too much noise)
        bin_cutoff = int(nbins/2)
        recovered_vals = []
        for value in fit_vals:
            fits_rec = value[:bin_cutoff]
            val_rec = np.mean(fits_rec)
                # average of the fit value in each bin up to cutoff
            recovered_vals.append(val_rec)
        
        recovered_values = {
            "m_fit_x" : recovered_vals[0],
            "m_fit_y" : recovered_vals[1],
            "m_fit_z" : recovered_vals[2],
            "b_fit" : recovered_vals[3]
        }

        # save recovered gradient values
        np.save(f"gradient_mocks/{grad_dim}D/patches/lst_sq_fit/recovered_vals_m-{m}-L_b-{b}_{n_patches}patches", recovered_values)

        # plot results
        plt.plot(r_avg, np.array(m_fits_x)/np.array(b_fits), color="black", marker=".", label="x fit")
        plt.plot(r_avg, np.array(m_fits_y)/np.array(b_fits), color="black", marker=".", alpha=0.6, label="y fit")
        plt.plot(r_avg, np.array(m_fits_z)/np.array(b_fits), color="black", marker=".", alpha=0.4, label="z fit")
        plt.vlines(r_avg[bin_cutoff], -0.05, 0.05, alpha=0.2, linestyle="dashed", label="Cutoff for grad calculation")
        plt.legend()

        fig1.savefig(f"gradient_mocks/{grad_dim}D/patches/lst_sq_fit/allbins_m-{m}-L_b-{b}_{n_patches}patches.png")