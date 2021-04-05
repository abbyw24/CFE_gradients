import numpy as np
import Corrfunc
from Corrfunc import bases, theory, utils, io
from matplotlib import pyplot as plt

######
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
print(x.shape, y.shape, z.shape)
print(min(x), max(x), min(y), max(y), min(z), max(z))
print(boxsize)

nd = len(mock_data)
nr = nd
######

# boxsize = 750.0
# x, y, z = io.read_lognormal_catalog(n='5e-5')
# nd = len(x)
# nr = nd
# print(x.shape, y.shape, z.shape)
# print(min(x), max(x), min(y), max(y), min(z), max(z))
# assert False

# random set
np.random.seed(1234)
x_rand = np.random.uniform(0, boxsize, nr)
y_rand = np.random.uniform(0, boxsize, nr)
z_rand = np.random.uniform(0, boxsize, nr)

# spline basis
proj_type = 'generalr'
kwargs = {'order': 3}
projfn = 'cubic_spline.dat'
rmin, rmax, ncomponents = 40.0, 152.0, 14
bases = bases.spline_bases(rmin, rmax, projfn, ncomponents, ncont=2000, **kwargs)

# computing projection vectors with DDsmu
r_edges = np.linspace(rmin, rmax, ncomponents+1)
nmubins = 1
mumax = 1.0
periodic = True
nthreads = 2

dd_res, dd_proj, _ = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                                  boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                                  ncomponents=ncomponents, projfn=projfn)
dr_res, dr_proj, _ = theory.DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z,
                                  X2=x_rand, Y2=y_rand, Z2=z_rand,
                                  boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                                  ncomponents=ncomponents, projfn=projfn)
rr_res, rr_proj, trr_proj = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins,
                                         x_rand, y_rand, z_rand, boxsize=boxsize,
                                         periodic=periodic, proj_type=proj_type,
                                         ncomponents=ncomponents, projfn=projfn)

# computing amplitudes
amps = utils.compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
r_fine = np.linspace(rmin, rmax, 2000)
xi_proj = utils.evaluate_xi(amps, r_fine, proj_type, projfn=projfn)

# plotting results with matplotlib
xi_res = theory.xi(boxsize, nthreads, r_edges, x, y, z, output_ravg=True)
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

plt.plot(r_avg, xi_standard, marker='o', ls='None', color='grey', label='Standard binned estimator')

plt.axhline(0.0, color='k', lw=1)
plt.xlim(min(r), max(r))
plt.ylim(-0.005, 0.025)
plt.xlabel(r'separation r ($h^{-1}$Mpc)')
plt.ylabel(r'$\xi$(r)')
plt.legend()
plt.show()

plt.savefig("gradient_mocks/"+str(grad_dim)+"D/suave/grad_xi_m-"+str(m)+"-L_b-"+str(b)+"_suave.png")