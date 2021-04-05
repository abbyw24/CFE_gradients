import numpy as np
import matplotlib.pyplot as plt
import math

######
# load in patch data
m = 0.25
b = 0.75
grad_dim = 1
n_patches = 8
L = np.load("boxsize.npy")

# and which bin we're using:
r_bin = 2
######

dim = ["x", "y", "z"]
patch_centers = np.load("gradient_mocks/"+str(grad_dim)+"D/patches/patch_centers/patch_centers_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.npy")
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
patches_xi = np.load("gradient_mocks/"+str(grad_dim)+"D/patches/grad_xi_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.npy",
                    allow_pickle=True)
r_avg, xi_patches, xi_patch_avg, xi_full = patches_xi

assert len(xi_patches) == n_patches

# clustering amplitudes
Y = xi_patches[:,r_bin-1]
    # 5th bin should be r ~ 35 h^-1 Mpc
r_avg_bin = r_avg[r_bin-1]*np.ones(n_patches)

# plot xi_patches
fig1 = plt.figure()
plt.title("Clustering amps in patches, m="+str(m)+", b="+str(b))
plt.xlabel(r"r ($h^{-1}$Mpc)")
plt.ylabel(r"$\xi$(r)")
cmap = plt.cm.get_cmap("cool")
ax = plt.axes()
ax.set_prop_cycle('color',cmap(np.linspace(0,1,n_patches)))
for patch in xi_patches:
    plt.plot(r_avg, patch, alpha=0.5, marker=".")
plt.scatter(r_avg_bin, Y, alpha=0.5, color="black")

# calculate matrix X = [b,m]
fig2 = plt.figure()
# color map– color code points to match corresponding patch center in grad_xi figure
plt.title("Linear least square fit, Clustering amps in patches (bin "+str(r_bin)+")")
plt.scatter(patch_centers[:,0], Y, marker="o", c=Y, cmap="cool", label="Mock: "+str(grad_dim)+"D, m = "+str(m)+", b = "+str(b))
plt.xlabel(r"Patch Centers ($h^{-1}$Mpc)")
plt.ylabel(r"$\xi$(r)")
x = np.linspace(min(patch_centers[:,0]),max(patch_centers[:,0]))

# set colors for best fit lines
bestfit_colors = ["blue","grey","silver"]
X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)
b_fit = X[0]
for i in range(len(dim)):
    plt.plot(x, X[i+1]*x + b_fit,color=bestfit_colors[i], label=dim[i]+" best fit: y = "+str("%.8f" %X[i+1])+"x + "+str("%.6f" %b_fit))

plt.legend()
plt.show()

# save figure
fig2.savefig("gradient_mocks/"+str(grad_dim)+"D/patches/lst_sq_fit/bin-"+str(r_bin)+"_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.png")