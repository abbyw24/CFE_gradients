import numpy as np
import matplotlib.pyplot as plt
import math
import Corrfunc

#####
# parameters for grad patches
grad_dim = 1
m = 1.0
b = 0.75
n_patches = 8
#####

# parameters for full grad
dim = "1D"
periodic = False
a = "m-"+str(m)+"-L_b-"+str(b)

r_avg_og, xi_original = np.load("gradient_mocks/"+dim+"/xi/grad_xi_"+a+"_per-"+str(periodic)+".npy")
xi_from_patches = np.load("gradient_mocks/"+str(grad_dim)+"D/patches/grad_xi_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.npy", allow_pickle=True)
r_avg_patches, xi_patches, xi_patch_avg, xi_full = xi_from_patches

# lognormal Corrfunc
lognorm_set = np.load("lognorm_set.npy")
x, y, z = lognorm_set[:,0], lognorm_set[:,1], lognorm_set[:,2]
nd = len(lognorm_set)
L = 2.0 * math.ceil(max(lognorm_set[:,0]))

nthreads = 1
periodic = False

rmin = 20.0
rmax = 150.0
nbins = 22
r_edges = np.linspace(rmin, rmax, nbins+1)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

# define L-S
def landy_szalay(nd, nr, dd, dr, rr):
    dd = dd/(nd*nd)
    dr = dr/(nd*nr)
    rr = rr/(nr*nr)
    xi_ls = (dd-2*dr+rr)/rr
    return xi_ls

# null set
nr = 2*nd
null_set = np.random.uniform(-L/2, L/2, (nr,3))
x_null, y_null, z_null = null_set[:,0], null_set[:,1], null_set[:,2]

# calculating Corrfunc for grad
dd_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x, y, z, boxsize=L, periodic=periodic)
dr_res_ln = Corrfunc.theory.DD(0, nthreads, r_edges, x, y, z, X2=x_null, Y2=y_null, Z2=z_null, boxsize=L, periodic=periodic)
rr_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=L, periodic=periodic)

# pull out only pair counts we need (dd,dr,rr) because Corrfunc.theory.DD returns a bunch of other stuff
dd_ln = np.array([x['npairs'] for x in dd_res_ln], dtype=float)
dr_ln = np.array([x['npairs'] for x in dr_res_ln], dtype=float)
rr_ln = np.array([x['npairs'] for x in rr_res_ln], dtype=float)

# calculating landy-szalay for grad, ln and dead
xi_ln = landy_szalay(nd,nr,dd_ln,dr_ln,rr_ln)

plt.plot(r_avg_og, xi_original,marker="o",label="original")
plt.plot(r_avg_patches, xi_full ,marker="o",label="from patches")
plt.plot(r_avg, xi_ln, marker="o",label="lognormal")

plt.legend()
plt.show()