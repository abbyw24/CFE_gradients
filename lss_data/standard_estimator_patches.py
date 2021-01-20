import numpy as np
import matplotlib.pyplot as plt
import math
import itertools as it
import Corrfunc

# load in split_mock and vector
split_mock = np.load("split_mock.npy")
v = np.load("split_mock_v.npy")
boxsize = 2.0 * math.ceil(max(split_mock[:,0]))

x, y, z = np.array([split_mock[:,0],split_mock[:,1],split_mock[:,2]])

boxsize_patch = boxsize/2

# load in lognormal and dead xis (for graphing)
split_xi = np.load("split_xi_per-True.npy")
r_avg_old = split_xi[0]
xi_split = split_xi[1]
xi_ln = split_xi[2]
xi_dead = split_xi[3]

# define octants
I_1 = (x>0) & (y>0) & (z>0)
I_2 = (x<0) & (y>0) & (z>0)
I_3 = (x<0) & (y<0) & (z>0)
I_4 = (x>0) & (y<0) & (z>0)
I_5 = (x>0) & (y>0) & (z<0)
I_6 = (x<0) & (y>0) & (z<0)
I_7 = (x<0) & (y<0) & (z<0)
I_8 = (x>0) & (y<0) & (z<0)
I = np.array([I_1,I_2,I_3,I_4,I_5,I_6,I_7,I_8])

# random set is calculated analytically with Corrfunc

# Corrfunc parameters
nthreads = 1
# the following is copied from split_corrfunc.py
rmin = 20.0
rmax = 100.0
nbins = 22
r_edges = np.linspace(rmin, rmax, nbins+1)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1]) 

# results for entire mock
# shift x,y,z to go from 0 - boxsize (instead of centered at 0)
x, y, z = np.array([x,y,z])+(boxsize/2)
results_xi = Corrfunc.theory.xi(boxsize, nthreads, r_edges, x, y, z)

# results in patches
xi = []
k = 0
fig = plt.figure()
for i in I:
    xs = split_mock[i]
    x, y, z = abs(np.array([xs[:,0],xs[:,1],xs[:,2]]))
    results_xi_patch = Corrfunc.theory.xi(boxsize_patch, nthreads, r_edges, x, y, z)
    plt.plot(r_avg,results_xi_patch["xi"],marker="o",alpha=0.5,label="Octant "+str(k+1))
    xi.append(results_xi_patch)
    k += 1
xi = np.array(xi)

# average of patch results
xi_patch_avg = np.sum(xi["xi"],axis=0)/len(xi)

# plot results
plt.plot(r_avg,results_xi["xi"],marker="o",color="black",label="Full Mock")
plt.plot(r_avg,xi_patch_avg,marker="o",color="black",alpha=0.5,label="Avg. of Patches")
plt.plot(r_avg_old,xi_ln,marker="o",color="darkblue",alpha=0.8,label="Lognormal")
plt.plot(r_avg_old,xi_dead,marker="o",color="darkblue",alpha=0.6,label="Dead")
# plot parameters
plt.xlabel("r")
plt.xlim(right=100)
plt.ylabel(r'$\xi$(r)')
plt.rcParams["axes.titlesize"] = 10
plt.title("Standard Estimator, Split Mock Patches, v="+str(v))
plt.legend()
plt.show()

# save plot
fig.savefig("split_patches.png")