import numpy as np
import matplotlib.pyplot as plt
import math
import Corrfunc

# load in lognormal, dead, and split mocks
lognorm_set = np.load("lognorm_set.npy")
dead_set = np.load("dead_set.npy")
split_mock = np.load("split_mock.npy")

# define N and boxsize (these are both values used in splitmock_gen.py but not loaded in)
N = len(lognorm_set)
boxsize = 2.0 * math.ceil(max(lognorm_set[:,0]))
    # boxsize = 2 * the max x value in lognorm_set, rounded UP to the nearest integer (returns a float though)
        # since the set is centered at x = 0, we need to multiply by 2 to get the full x-range/boxsize

# CALCULATING CORRFUNC
# define NULL (random) set for Corrfunc
nr = 2*N
x_null = boxsize * (np.random.rand(nr) - .5)
y_null = boxsize * (np.random.rand(nr) - .5)
z_null = boxsize * (np.random.rand(nr) - .5)
null_set = np.array([x_null,y_null,z_null]).T

# relabeling mocks / pulling out x, y, z values
x_split = split_mock[:,0]
y_split = split_mock[:,1]
z_split = split_mock[:,2]

x_lognorm = lognorm_set[:,0]
y_lognorm = lognorm_set[:,1]
z_lognorm = lognorm_set[:,2]

x_dead = dead_set[:,0]
y_dead = dead_set[:,1]
z_dead = dead_set[:,2]

# parameters
nthreads = 1
periodic = False
nd = N

rmin = 20.0
rmax = 150.0
nbins = 22
r_edges = np.linspace(rmin, rmax, nbins+1)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

# define Landy-Szalay
def landy_szalay(nd, nr, dd, dr, rr):
    dd = dd/(nd*nd)
    dr = dr/(nd*nr)
    rr = rr/(nr*nr)
    xi_ls = (dd-2*dr+rr)/rr
    return xi_ls

# calculating Corrfunc for split
dd_res_split = Corrfunc.theory.DD(1, nthreads, r_edges, x_split, y_split, z_split, boxsize=boxsize, periodic=periodic)
dr_res_split = Corrfunc.theory.DD(0, nthreads, r_edges, x_split, y_split, z_split, X2=x_null, Y2=y_null, Z2=z_null, boxsize=boxsize, periodic=periodic)
rr_res_split = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=boxsize, periodic=periodic)

# pull out only pair counts we need (dd,dr,rr) because Corrfunc.theory.DD returns a bunch of other stuff
dd_split = np.array([x['npairs'] for x in dd_res_split], dtype=float)
dr_split = np.array([x['npairs'] for x in dr_res_split], dtype=float)
rr_split = np.array([x['npairs'] for x in rr_res_split], dtype=float)

# calculating Corrfunc for lognormal mock (ln = lognormal)
dd_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_lognorm, y_lognorm, z_lognorm, boxsize=boxsize, periodic=periodic)
dr_res_ln = Corrfunc.theory.DD(0, nthreads, r_edges, x_lognorm, y_lognorm, z_lognorm, X2=x_null, Y2=y_null, Z2=z_null, boxsize=boxsize, periodic=periodic)
rr_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=boxsize, periodic=periodic)

dd_ln = np.array([x['npairs'] for x in dd_res_ln], dtype=float)
dr_ln = np.array([x['npairs'] for x in dr_res_ln], dtype=float)
rr_ln = np.array([x['npairs'] for x in rr_res_ln], dtype=float)

# calculating Corrfunc for "dead" catalog
dd_res_dead = Corrfunc.theory.DD(1, nthreads, r_edges, x_dead, y_dead, z_dead, boxsize=boxsize, periodic=periodic)
dr_res_dead = Corrfunc.theory.DD(0, nthreads, r_edges, x_lognorm, y_lognorm, z_lognorm, X2=x_null, Y2=y_null, Z2=z_null, boxsize=boxsize, periodic=periodic)
rr_res_dead = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=boxsize, periodic=periodic)

dd_dead = np.array([x['npairs'] for x in dd_res_dead], dtype=float)
dr_dead = np.array([x['npairs'] for x in dr_res_dead], dtype=float)
rr_dead = np.array([x['npairs'] for x in rr_res_dead], dtype=float)

# calculating landy-szalay for split, ln and dead
xi_split = landy_szalay(nd,nr,dd_split,dr_split,rr_split)
xi_ln = landy_szalay(nd,nr,dd_ln,dr_ln,rr_ln)
xi_dead = landy_szalay(nd,nr,dd_dead,dr_dead,rr_dead)

# combine corr funcs for split, lognormal, and dead into one array for saving
split_xi = np.array([r_avg,xi_split,xi_ln,xi_dead])

# save Corrfunc data
np.save("split_xi",split_xi)