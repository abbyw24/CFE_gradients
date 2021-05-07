import os
import numpy as np
import matplotlib.pyplot as plt

import Corrfunc
from Corrfunc.io import read_lognormal_catalog
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.xi import xi
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import trr_analytic
from Corrfunc.bases import bao_bases
from colossus.cosmology import cosmology

# load in data
m = 0.5
b = 0.5
grad_dim = 1
boxsize = np.load("boxsize.npy")

mock_data = np.load("gradient_mocks/"+str(grad_dim)+"D/mocks/grad_mock_m-"+str(m)+"-L_b-"+str(b)+".npy")
mock_data += boxsize/2

x = mock_data[:,0]
y = mock_data[:,1]
z = mock_data[:,2]

nd = len(x)

# construct bao basis
rmin = 40
rmax = 150
cosmo = cosmology.setCosmology('planck15')
redshift = 1.0
bias = 2.0
alpha_guess = 1.0
k0 = 0.1
projfn = 'bao_basis.dat'
bases = bao_bases(rmin, rmax, projfn, cosmo_base=cosmo, alpha_guess=alpha_guess, k0=k0, 
                  ncont=2000, redshift=0.0, bias=1.0)

# plot bao basis
plt.figure(figsize=(8,5))
bao_base_colors = ['#41ab5d', '#74c476', '#a1d99b', '#005a32', '#238b45'] #from https://colorbrewer2.org/#type=sequential&scheme=Greens&n=8, last 5 out of 8
bao_base_names = [r'$\frac{k_1}{s^2}$', r'$\frac{k_2}{s}$', r'$k_3$', 
                  r'$\xi^\mathrm{mod}(\alpha_\mathrm{guess} s)$', 
                  r'$k_0 \frac{\mathrm{d} \xi^\mathrm{mod}(\alpha_\mathrm{guess} s)}{\mathrm{d} \alpha}$']
r = bases[:,0]
base_vals = bases[:,1:]
for i in range(base_vals.shape[1]):
    plt.plot(r, base_vals[:,i], label=bao_base_names[i], color=bao_base_colors[i])
    
plt.legend()
plt.xlim(rmin, rmax)
plt.ylim(-0.0025, 0.01)
plt.xlabel(r'separation $r$ ($h^{-1}\,$Mpc)')
plt.ylabel('BAO basis functions $f_k(r)$')

# suave with bao basis
nthreads = 1
# Need to give a dummy r_edges for compatibility with standard Corrfunc.
# But we will use this later to compute the standard xi, so give something reasonable.
r_edges = np.linspace(rmin, rmax, 15)
mumax = 1.0
nmubins = 1
periodic = True
proj_type = 'generalr'
ncomponents = base_vals.shape[1]

# calcualting dd terms
dd_res_bao, dd_bao, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, 
                           boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                           ncomponents=ncomponents, projfn=projfn)
# calculate other terms analytically
volume = boxsize**3
rr_ana_bao, trr_ana_bao = trr_analytic(rmin, rmax, nd, volume, ncomponents, proj_type, projfn=projfn)

numerator = dd_bao - rr_ana_bao
amps_ana_bao = np.linalg.solve(trr_ana_bao, numerator)

# calculate xi
r_fine = np.linspace(rmin, rmax, 2000)
xi_ana_bao = evaluate_xi(amps_ana_bao, r_fine, proj_type, projfn=projfn)

# standard correlation function for comparison
xi_res = xi(boxsize, nthreads, r_edges, x, y, z, output_ravg=True)
r_avg, xi_standard = xi_res['ravg'], xi_res['xi']

# plot results
plt.figure(figsize=(8,5))
plt.plot(r_fine, xi_ana_bao, color='green', label='BAO basis')
plt.plot(r_avg, xi_standard, marker='o', ls='None', color='grey', label='Standard estimator')
plt.xlim(rmin, rmax)
plt.xlabel(r'r ($h^{-1}$Mpc)')
plt.ylabel(r'$\xi$(r)')
plt.legend()
plt.show()