import numpy as np
import matplotlib.pyplot as plt
import math
import itertools as it
import Corrfunc

# load in split_mock
split_mock = np.load("split_mock.npy")
x = split_mock[:,0]
y = split_mock[:,1]
z = split_mock[:,2]
print(split_mock.shape)

# define octants
I_1 = (x>0) & (y>0) & (z>0)
I_2 = (x<0) & (y>0) & (z>0)
I_3 = (x<0) & (y<0) & (z>0)
I_4 = (x>0) & (y<0) & (z>0)
I_5 = (x>0) & (y>0) & (z<0)
I_6 = (x<0) & (y>0) & (z<0)
I_7 = (x<0) & (y<0) & (z<0)
I_8 = (x>0) & (y<0) & (z<0)
# divide split mock into octants
split1 = split_mock[I_1]
split2 = split_mock[I_2]
split3 = split_mock[I_3]
split4 = split_mock[I_4]
split5 = split_mock[I_5]
split6 = split_mock[I_6]
split7 = split_mock[I_7]
split8 = split_mock[I_8]

# Corrfunc parameters
boxsize = 2.0 * math.ceil(max(x))
print(boxsize)
nthreads = 1
# the following is copied from split_corrfunc.py
rmin = 20.0
rmax = 150.0
nbins = 22
r_edges = np.linspace(rmin, rmax, nbins+1)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1]) 

# results
results_xi = Corrfunc.theory.xi(boxsize, nthreads, r_edges, x, y, z)

# plot results
plt.plot(r_avg,results_xi)

# dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins,
#                            x, y, z, boxsize=boxsize, periodic=periodic,
#                            proj_type=proj_type, nprojbins=nprojbins)