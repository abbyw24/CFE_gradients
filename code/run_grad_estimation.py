import numpy as np
import matplotlib.pyplot as plt
import os

import read_lognormal
from corrfunc_ls import xi

import globals

globals.initialize_vals()  # brings in all the default parameters

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

clust_val = 2

Lx, Ly, Lz, nd, data = read_lognormal.read(f"/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy_As{clust_val}x/cat_L750_n2e-4_z057_patchy_As{clust_val}x_lognormal_rlz0.bin")
data = data.T   # transpose to fit requirements for xi function
L = Lx  # boxsize

# if there are negative values, shift by L/2, to 0 to L
if np.any(data <= 0):
    data += L/2
else:
    assert np.all(data >= 0 and data <= L)

# CORRFUNC
# random set
nr = randmult*nd
rand_set = np.random.uniform(0, L, (nr,3))

# results
results_xi = xi(data, rand_set, periodic, nthreads, rmin, rmax, nbins)
r_avg = results_xi[0]
xi = np.array(results_xi[1])
print(r_avg, xi)

# plot results
fig, ax = plt.subplots()
plt.plot(r_avg, xi, color="black", marker=".", label="Full Mock")
ax.set_xlabel(r'r ($h^{-1}$Mpc)')
ax.set_ylabel(r'$\xi$(r)')
ax.set_title(f"Standard Estimator, {clust_val}x Lognormal Mock")

fig.savefig(f"/scratch/aew492/research-summer2020_output/Corrfunc_{clust_val}x")

# SUAVE
