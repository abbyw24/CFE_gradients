import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123456)

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.xi import xi
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic
from Corrfunc.bases import spline

# define function to generate a random set of points in xyz
def rand_set(n):
    x = boxsize * np.random.rand(n)
    y = boxsize * np.random.rand(n)
    z = boxsize * np.random.rand(n)
    rand_set = np.array([x,y,z]).T
    return(rand_set)

# important parameters
boxsize = 20.0
bins=15
bin_edges = np.linspace(0.01,boxsize,bins+1)
    # starting bins at 0.01 instead of exactly 0 to avoid pair counting
    #   problems later on (to match Kate's Corrfunc)

# GENERATING DATA SET ("DD")
#   both sets are random but we will designate this one as the actual data set
set1 = []
set1 = rand_set(50) # 50 data points / "galaxies"

# calculating distances between all points
distances1=[]
for i in set1:
    for j in set1:
        d = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2)
        distances1.append(d)
        #plt.plot([i[0],j[0]],[i[1],j[1]])
            #plotting was helpful for small 2D data sets, but not anymoreee

# binning distances
DD, bins1, _ = plt.hist(distances1, bins=bin_edges, color="black", label="DD")
    # fyi, plt.hist() returns n=values of hist bins, bins=bin edges, patches

# GENERATING RANDOM SET
set2 = []
set2 = rand_set(80)

distances2=[]
for i in set2:
    for j in set2:
        d = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2)
        distances2.append(d)
        #plt.plot([i[0],j[0]],[i[1],j[1]])

RR, bins2, _ = plt.hist(distances2, bins=bin_edges, alpha=0.6, color="blue", label="RR")

# GENERATING DR
distances_dr=[]
for i in set1:
    for j in set2:
        d = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2)
        distances_dr.append(d)

DR, bins_dr, _ = plt.hist(distances_dr, bins=bin_edges, alpha=0.6, color="green", label="DR")
# graph formatting
plt.title("Distances")
plt.legend()

# CALCULATING CORRELATION FUNCTION MANUALLY
# more parameters
nd = len(set1)
nr = len(set2)

# define Landy-Szalay estimator
def landy_szalay(nd, nr, dd, dr, rr):
    print("dd=",dd)
    dd = dd/(nd*nd)
    dr = dr/(nd*nr)
    print("dr=",dr)
    rr = rr/(nr*nr)
    print("rr=",rr)
    xi_ls = (dd-2*dr+rr)/rr
    return xi_ls

xi_ls_man = landy_szalay(nd,nr,DD,DR,RR)

# creating array of bin centers
bins_avg=[]
for i in range(len(bin_edges)-1):
    avg = (bin_edges[i+1]+bin_edges[i])/2
    bins_avg.append(avg)

# plotting manual correlation function
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.scatter(bins_avg,xi_ls_man,label="Manual")
ax1.set_xlabel(r'r')
ax1.set_ylabel(r'$\xi$(r)')
ax1.set_title(r'"Manual" $\xi(r)$')

# NATIVE CORRFUNC
# more parameters! this cell is needed for all Corrfunc functions! / not just Landy-Szalay
x, y, z = set1[:,0], set1[:,1], set1[:,2]
x_rand, y_rand, z_rand = set2[:,0], set2[:,1], set2[:,2]

rmin = 0.0
rmax = 20.0 # to match boxsize
nbins = len(bin_edges)-1
r_avg = 0.5*(bin_edges[1:]+bin_edges[:-1])
r_fine = np.linspace(min(bin_edges), max(bin_edges), 1000)

periodic = False # because this data set does not wrap around values across the volume
nthreads = 1

dd_res = Corrfunc.theory.DD(1, nthreads, bin_edges, x, y, z, boxsize=boxsize, periodic=periodic)
dr_res = Corrfunc.theory.DD(0, nthreads, bin_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsize, periodic=periodic)
rr_res = Corrfunc.theory.DD(1, nthreads, bin_edges, x_rand, y_rand, z_rand, boxsize=boxsize, periodic=periodic)

dd = np.array([x['npairs'] for x in dd_res], dtype=float)
dr = np.array([x['npairs'] for x in dr_res], dtype=float)
rr = np.array([x['npairs'] for x in rr_res], dtype=float)

xi_ls = landy_szalay(nd,nr,dd,dr,rr)

# plot native Corrfunc
ax2 = fig.add_subplot(221)
ax2.plot(r_avg, xi_ls,'o',label="Standard")
ax2.set_xlabel(r'r ($h^{-1}$Mpc)')
ax2.set_ylabel(r'$\xi$(r)')
ax2.set_title(r"Landy-Szalay")

plt.show()