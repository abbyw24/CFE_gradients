import numpy as np
import matplotlib.pyplot as plt
import math

######
# load in patch data
m = 0.5
b = 0.5
grad_dim = 1
n_patches = 8
nbins = 22
L = np.load("boxsize.npy")

dim = ["x", "y", "z"]
patch_centers = np.load("gradient_mocks/"+str(grad_dim)+"D/patches/patch_centers/patch_centers_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.npy")
patch_centers = patch_centers - L/2
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
patches_xi = np.load("gradient_mocks/"+str(grad_dim)+"D/patches/grad_xi_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.npy",
                    allow_pickle=True)
r_avg, xi_patches, xi_patch_avg, xi_full = patches_xi

assert len(xi_patches) == n_patches

# plot xi_patches
fig1 = plt.figure()
plt.title("Clustering amps in patches, m="+str(m)+", b="+str(b))
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

m_fits_x = []
m_fits_y = []
m_fits_z = []
b_fits = []
for r_bin in range(nbins):
    # clustering amplitudes
    Y = xi_patches[:,r_bin]
    # least square fit
    X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)
    m_fits_x.append(X[1])
    m_fits_y.append(X[2])
    m_fits_z.append(X[3])
    b_fits.append(X[0])
fit_vals = [m_fits_x,m_fits_y,m_fits_z,b_fits]

# create our recovered gradient array (as of now with a set n_bin cutoff to avoid too much noise)
bin_cutoff = int(nbins/2)
recovered_vals = []
for value in fit_vals:
    fits_rec = value[:bin_cutoff]
    val_rec = np.mean(fits_rec)
    recovered_vals.append(val_rec)
print(recovered_vals)
# save recovered gradient values
np.save("gradient_mocks/"+str(grad_dim)+"D/patches/lst_sq_fit/recovered_vals_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches",recovered_vals)

# plot results
plt.plot(r_avg, np.array(m_fits_x)/np.array(b_fits), color="black", marker=".", label="x fit")
plt.plot(r_avg, np.array(m_fits_y)/np.array(b_fits), color="black", marker=".", alpha=0.6, label="y fit")
plt.plot(r_avg, np.array(m_fits_z)/np.array(b_fits), color="black", marker=".", alpha=0.4, label="z fit")
plt.vlines(r_avg[bin_cutoff], -0.05, 0.05, alpha=0.2, linestyle="dashed", label="Cutoff for grad calculation")

plt.legend()
plt.show()

fig1.savefig("gradient_mocks/"+str(grad_dim)+"D/patches/lst_sq_fit/allbins_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.png")