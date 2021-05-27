import numpy as np
import matplotlib.pyplot as plt
import os

import Corrfunc
from Corrfunc import bases, theory, utils, io

from suave import cosmo_bases
from create_subdirs import create_subdirs
import read_lognormal
from corrfunc_ls import xi_ls

import globals

globals.initialize_vals()  # brings in all the default parameters

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

clust_val = 2

Lx, Ly, Lz, nd, ln_data = read_lognormal.read(f"/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy_As{clust_val}x/cat_L750_n2e-4_z057_patchy_As{clust_val}x_lognormal_rlz0.bin")
L = Lx  # boxsize
x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = ln_data.T
data_set = (np.array([x_lognorm, y_lognorm, z_lognorm]))
x, y, z = data_set
print(data_set)
print(len(x))

# create necessary subdirectories
create_subdirs("/scratch/aew492/research-summer2020_output/", ["lognormal"])

# if there are negative values, shift by L/2, to 0 to L
if np.any(data_set <= 0):
    data_set += L/2
else:
    assert np.all(data_set >= 0 and data_set <= L)

# random set
nr = randmult*nd
rand_set = np.random.uniform(0, L, (3, nr))
print(rand_set)
x_rand, y_rand, z_rand = rand_set
print(len(x_rand))

# CORRFUNC
# results
results_xi = xi_ls(data_set, rand_set, periodic, nthreads, rmin, rmax, nbins)
r_avg = results_xi[0]
xi_standard = np.array(results_xi[1])
print(xi_standard)

# plot results
fig1, ax1 = plt.subplots()
plt.plot(r_avg, xi_standard, color="black", marker=".", label="Full Mock")
ax1.set_xlabel(r'r ($h^{-1}$Mpc)')
ax1.set_ylabel(r'$\xi$(r)')
ax1.set_title(f"Standard Estimator, {clust_val}x Lognormal Mock")

fig1.savefig(f"/scratch/aew492/research-summer2020_output/lognormal/Corrfunc_{clust_val}x")
print("Corrfunc done")

# SUAVE
# parameters:
# spline basis
proj_type = 'generalr'
kwargs = {'order': 3}
projfn = 'cubic_spline.dat'
ncomponents = 14    # what should this be?
bases = bases.spline_bases(rmin, rmax, projfn, ncomponents, ncont=2000, **kwargs)
print("bases done")
# computing projection vectors with DDsmu
rmin = rmin
rmax = rmax
r_edges = np.linspace(rmin, rmax, ncomponents+1)
r_fine = np.linspace(rmin, rmax, 2000)
nmubins = 1
mumax = 1.0

# run the pair counts
dd_res, dd_proj, _ = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                        proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, periodic=periodic)
print("DD:", np.array(dd_proj))

dr_res, dr_proj, _ = theory.DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand,
                        proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, periodic=periodic)
print("DR:", np.array(dr_proj))

rr_res, rr_proj, qq_proj = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins, x_rand, y_rand, z_rand, 
                                proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, periodic=periodic)
print("RR:", np.array(rr_proj))

# computing amplitudes
amps = utils.compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
print("compute amps done")

xi_proj = utils.evaluate_xi(amps, r_fine, proj_type, projfn=projfn)
print("xi_proj done")

# plotting results with matplotlib
xi_res = theory.xi(L, nthreads, r_edges, x, y, z, output_ravg=True)
print("xi_res done")
r_avg, xi_standard = xi_res['ravg'], xi_res['xi']

plt.figure(figsize=(10,7))
plt.plot(r_fine, xi_proj, color='red', lw=1.5, label='Continuous-Function Estimator with spline basis')

r = bases[:,0]
base_vals = bases[:,1:]
for i in range(base_vals.shape[1]):
    label = None
    if i==0:
        label = 'Cubic spline basis functions'
    plt.plot(r, amps[i]*base_vals[:,i], color='darkred', lw=0.5, label=label)
    print(f"loop plot {i} done")

plt.plot(r_avg, xi_standard, marker='o', ls='None', color='grey', label='Standard binned estimator')

plt.axhline(0.0, color='k', lw=1)
plt.xlim(min(r), max(r))
plt.ylim(-0.005, 0.025)
plt.xlabel(r'separation r ($h^{-1}$Mpc)')
plt.ylabel(r'$\xi$(r)')
plt.legend()

plt.savefig(f"/scratch/aew492/research-summer2020_output/lognormal/suave_{clust_val}x")