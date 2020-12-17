import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
import Corrfunc

# pick a seed number so that random set stays the same every time
np.random.seed(123456)

# LOGNORMAL SET
# read in mock data
Lx, Ly, Lz, N, data = read_lognormal.read('/Users/abbywilliams/Physics/research/lss_data/lognormal_mocks/cat_L750_n3e-4_lognormal_rlz0.bin')
    # define boxsize based on mock; and N = number of data points
boxsize = Lx

x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
lognorm_set = x_lognorm, y_lognorm, z_lognorm = np.array([x_lognorm, y_lognorm, z_lognorm])-(boxsize/2)

# DEAD SET
# generate a random data set (same size as mock)
x_dead = boxsize * (np.random.rand(N) - .5)
y_dead = boxsize * (np.random.rand(N) - .5)
z_dead = boxsize * (np.random.rand(N) - .5)
dead_set = np.array([x_dead,y_dead,z_dead])
print(min(x_dead),max(x_dead))

# generate random R3 unit vector
v = np.random.normal(size=3)
v[2] = 0
v_len = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
v = v / v_len
print(v)
print(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2))

# # define dividing plane normal to vector v
# #   a*x + b*y + c*z = 0 ==> z = -(a*x + b*y) / c
# x, y = np.meshgrid(range(int(-boxsize/2),int(boxsize/2)),range(int(-boxsize/2),int(boxsize/2)))
# z = (v[0]*x + v[1]*y) # / v[2]
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(x, y, z)

# CREATE NEW MOCK DIVIDED ALONG VECTOR LINE
# now we want one side of the plane to be the lognormal mock, and the other side to be the normal mock
# define dot products of vector v dot lognormal mock, and vector v dot random mock
lognorm_dot = np.dot(v,lognorm_set)
dead_dot = np.dot(v,dead_set)

# define the lognormal part of the new mock as where v dot random
lognorm_part = lognorm_set.T[np.where(lognorm_dot > 0)]
dead_part = dead_set.T[np.where(dead_dot < 0)]

# combine lognorm_part and dead_part in a new mock
new_mock = []
new_mock = np.append(lognorm_part,dead_part,axis=0)

# xy-slice to visualize new mock!
z_max = 100
xy_slice = new_mock[np.where(dead_set[2] < z_max)] # select rows where z < z_max

plt.figure()
plt.plot(xy_slice[:,0],xy_slice[:,1],',',label=v)   # plot scatter xy-slice
plt.plot(xy_slice[:,0],(v[1]/v[0])*xy_slice[:,0])   # plot vector v (no z)
plt.plot(xy_slice[:,0],-(v[0]/v[1])*xy_slice[:,0])  # plot x, y of dividing plane (normal to v)
plt.axes().set_aspect("equal")                      # square aspect ratio
plt.ylim((-400,400))
plt.xlabel("x (Mpc/h)")
plt.ylabel("y (Mpc/h)")
plt.title("Split Mock")

# CALCULATING CORRFUNC
# define NULL (random) set for Corrfunc
nr = 2*N
x_null = boxsize * (np.random.rand(nr) - .5)
y_null = boxsize * (np.random.rand(nr) - .5)
z_null = boxsize * (np.random.rand(nr) - .5)
null_set = np.array([x_null,y_null,z_null]).T

# relabeling mocks
x_split = new_mock[:,0]
y_split = new_mock[:,1]
z_split = new_mock[:,2]

# parameters
nthreads = 1
periodic = True
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
split_data = np.array([r_avg,xi_split,xi_ln,xi_dead])

# save Corrfunc data
np.save("split_data",split_data)