import numpy as np
import matplotlib.pyplot as plt
import os

import Corrfunc
from Corrfunc import bases, theory, utils, io

from suave import cosmo_bases
from create_subdirs import create_subdirs
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
x, y, z = data[:,0], data[:,1], data[:,2]
L = Lx  # boxsize

# create necessary subdirectories
create_subdirs("/scratch/aew492/research-summer2020_output/", ["lognormal"])

# if there are negative values, shift by L/2, to 0 to L
if np.any(data <= 0):
    data += L/2
else:
    assert np.all(data >= 0 and data <= L)

# random set
nr = randmult*nd
rand_set = np.random.uniform(0, L, (nr,3))
x_rand, y_rand, z_rand = rand_set[:,0], rand_set[:,1], rand_set[:,2]


# CORRFUNC
# results
results_xi = xi(data, rand_set, periodic, nthreads, rmin, rmax, nbins)
r_avg = results_xi[0]
xi = np.array(results_xi[1])

# plot results
fig1, ax1 = plt.subplots()
plt.plot(r_avg, xi, color="black", marker=".", label="Full Mock")
ax1.set_xlabel(r'r ($h^{-1}$Mpc)')
ax1.set_ylabel(r'$\xi$(r)')
ax1.set_title(f"Standard Estimator, {clust_val}x Lognormal Mock")

fig1.savefig(f"/scratch/aew492/research-summer2020_output/lognormal/Corrfunc_{clust_val}x")


# SUAVE
# spline basis
proj_type = 'generalr'
kwargs = {'order': 3}
projfn = 'cubic_spline.dat'
bases = cosmo_bases(rmin, rmax, projfn)
ncomponents = 4*(bases.shape[1]-1)

# computing projection vectors with DDsmu
r_edges = np.linspace(rmin, rmax, ncomponents+1)
nmubins = 1
mumax = 1.0

dd_res, dd_proj, _ = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                                  boxsize=L, periodic=periodic, proj_type=proj_type,
                                  ncomponents=ncomponents, projfn=projfn)
dr_res, dr_proj, _ = theory.DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z,
                                  X2=x_rand, Y2=y_rand, Z2=z_rand,
                                  boxsize=L, periodic=periodic, proj_type=proj_type,
                                  ncomponents=ncomponents, projfn=projfn)
rr_res, rr_proj, trr_proj = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins,
                                         x_rand, y_rand, z_rand, boxsize=L,
                                         periodic=periodic, proj_type=proj_type,
                                         ncomponents=ncomponents, projfn=projfn)

# computing amplitudes
amps = utils.compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
r_fine = np.linspace(rmin, rmax, 2000)
xi_proj = utils.evaluate_xi(amps, r_fine, proj_type, projfn=projfn)

# plotting results with matplotlib
xi_res = theory.xi(L, nthreads, r_edges, x, y, z, output_ravg=True)
r_avg, xi_standard = xi_res['ravg'], xi_res['xi']

fig2, ax2 = plt.subplots()
plt.plot(r_fine, xi_proj, color='red', lw=1.5, label='Continuous-Function Estimator with spline basis')

r = bases[:,0]
base_vals = bases[:,1:]
for i in range(base_vals.shape[1]):
    label = None
    if i==0:
        label = 'Cubic spline basis functions'
    plt.plot(r, amps[i]*base_vals[:,i], color='darkred', lw=0.5, label=label)

plt.plot(r_avg, xi_standard, marker='o', ls='None', color='grey', label='Standard binned estimator')

plt.axhline(0.0, color='k', lw=1)
ax2.set_xlim(min(r), max(r))
ax2.set_ylim(-0.005, 0.025)
ax2.set_xlabel(r'separation r ($h^{-1}$Mpc)')
ax2.set_ylabel(r'$\xi$(r)')
plt.legend()

fig2.savefig(f"/scratch/aew492/research-summer2020_output/lognormal/suave_{clust_val}x")