import numpy as np 
import matplotlib.pyplot as plt 
import read_lognormal
import Corrfunc

# pick a seed number so that random set stays the same every time
np.random.seed(1234567)

# read in mock data
Lx, Ly, Lz, N, data = read_lognormal.read('/Users/abbywilliams/Physics/research/lss_data/lognormal_mocks/cat_L750_n3e-4_lognormal_rlz0.bin')
x_ln, y_ln, z_ln, vx_ln, vy_ln, vz_ln = data.T

# define boxsize based on mock; and N = number of data points
boxsize = Lx

# generate a random data set (same size as mock)
x_rand = boxsize * np.random.rand(N)
y_rand = boxsize * np.random.rand(N)
z_rand = boxsize * np.random.rand(N)
rand_set = np.array([x_rand,y_rand,z_rand]).T

plt.show()

# "HALF AND HALF"
# now we want half of our new mock to be the read-in data set, and half to be the random set
#   say, if x < (boxsize/2), then data=mock and if x >= (boxsize/2), then data=random

# locate all values in MOCK for which x < (boxsize/2)
where_halfx_ln = np.where(x_ln < (boxsize/2))
halfx_ln = x_ln[where_halfx_ln]
# find associated y and z values (so all 3 arrays are the same size)
halfy_ln = y_ln[where_halfx_ln]
halfz_ln = z_ln[where_halfx_ln]
# combine x, y, and z values into one array:
half_ln = np.array([halfx_ln,halfy_ln,halfz_ln]).T

# locate all values in RANDOM set for which x >= (boxsize/2)
where_halfx_rand = np.where(x_rand >= (boxsize/2))
halfx_rand = x_rand[where_halfx_rand]
# find associated y and z values (so all 3 arrays are the same size)
halfy_rand = y_rand[where_halfx_rand]
halfz_rand = z_rand[where_halfx_rand]
# combine x, y, and z values into one array:
half_rand = np.array([halfx_rand,halfy_rand,halfz_rand]).T

# create new mock with half lognormal mock and half random set
new_mock = []
new_mock = np.append(half_ln,half_rand,axis=0)
x = new_mock[:,0]
y = new_mock[:,1]
z = new_mock[:,2]
print(new_mock)

# xy-slice to visualize new mock!
z_max = 100
xy_slice = new_mock[z < z_max] # select rows where z < z_max
plt.plot(xy_slice[:,0],xy_slice[:,1],',')
plt.xlabel("x (Mpc/h)")
plt.ylabel("y (Mpc/h)")
plt.title('"Half and Half" Mock')

# define random set for Corrfunc
    # note! x_rand etc. were already defined earlier in creating the random set for new_mock;
        # but this should be fine for now
nr = 2*N
x_rand = boxsize * np.random.rand(nr)
y_rand = boxsize * np.random.rand(nr)
z_rand = boxsize * np.random.rand(nr)
rand_set = np.array([x_rand,y_rand,z_rand]).T

# CALCULATING CORRFUNC
# parameters
nthreads = 1
periodic = False
nd = N
# these are from the demo notebook so not sure about these:
rmin = 20.0 #40.0
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

# calculating Corrfunc for new_mock (nm = new_mock)
dd_res_nm = Corrfunc.theory.DD(1, nthreads, r_edges, x, y, z, boxsize=boxsize, periodic=periodic)
dr_res_nm = Corrfunc.theory.DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsize, periodic=periodic)
rr_res_nm = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, boxsize=boxsize, periodic=periodic)

# pull out only pair counts we need (dd,dr,rr) because Corrfunc.theory.DD returns a bunch of other stuff
dd_nm = np.array([x['npairs'] for x in dd_res_nm], dtype=float)
dr_nm = np.array([x['npairs'] for x in dr_res_nm], dtype=float)
rr_nm = np.array([x['npairs'] for x in rr_res_nm], dtype=float)

# calculating Corrfunc for lognormal mock (ln = lognormal)
dd_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_ln, y_ln, z_ln, boxsize=boxsize, periodic=periodic)
dr_res_ln = Corrfunc.theory.DD(0, nthreads, r_edges, x_ln, y_ln, z_ln, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsize, periodic=periodic)
rr_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, boxsize=boxsize, periodic=periodic)

dd_ln = np.array([x['npairs'] for x in dd_res_ln], dtype=float)
dr_ln = np.array([x['npairs'] for x in dr_res_ln], dtype=float)
rr_ln = np.array([x['npairs'] for x in rr_res_ln], dtype=float)

# calculating landy-szalay for nm and ln
xi_nm = landy_szalay(nd,nr,dd_nm,dr_nm,rr_nm)
xi_ln = landy_szalay(nd,nr,dd_ln,dr_ln,rr_ln)

# plot Corrfunc for new_mock
plt.figure()
plt.plot(r_avg, xi_nm, marker='o', label="Half and half")
plt.xlabel(r'r ($h^{-1}$Mpc)')
plt.ylabel(r'$\xi$(r)')
plt.title(r"Landy-Szalay")

# plot Corrfunc for lognormal mock
plt.plot(r_avg, xi_ln, marker='o', label="Lognormal")

plt.legend()
plt.show()

