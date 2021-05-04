import numpy as np
import matplotlib.pyplot as plt
import math
import Corrfunc
import globals

globals.initialize_vals()

grad_dim = globals.grad_dim
L = globals.L
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

r_edges = np.linspace(rmin, rmax, nbins+1)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

# define Landy-Szalay
def landy_szalay(nd, nr, dd, dr, rr):
    dd = dd/(nd*nd)
    dr = dr/(nd*nr)
    rr = rr/(nr*nr)
    xi_ls = (dd-2*dr+rr)/rr
    return xi_ls

# load in DEAD and LOGNORMAL sets
dead_set = np.load("dead_set.npy")
x_dead, y_dead, z_dead = dead_set.T

lognormal_set = np.load("lognormal_set.npy")
x_lognorm, y_lognorm, z_lognorm = lognormal_set.T

# make sure dead set and lognormal set are the same size (for purposes of defining N)
assert len(dead_set) == len(lognormal_set)

# define N
nd = len(lognormal_set)

# define NULL (random) set for Corrfunc
nr = randmult*nd
null_set = np.random.uniform(-L/2, L/2, (nr,3))
x_null, y_null, z_null = null_set.T

# calculating corrfunc for DEAD and LOGNORMAL catalogs
# calculating corrfunc for "dead" (random) catalog
dd_res_dead = Corrfunc.theory.DD(1, nthreads, r_edges, x_dead, y_dead, z_dead, boxsize=L, periodic=periodic)
dr_res_dead = Corrfunc.theory.DD(0, nthreads, r_edges, x_dead, y_dead, z_dead, X2=x_null, Y2=y_null, Z2=z_null, boxsize=L, periodic=periodic)
rr_res_dead = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=L, periodic=periodic)

dd_dead = np.array([x['npairs'] for x in dd_res_dead], dtype=float)
dr_dead = np.array([x['npairs'] for x in dr_res_dead], dtype=float)
rr_dead = np.array([x['npairs'] for x in rr_res_dead], dtype=float)

# calculating corrfunc for lognormal mock (ln = lognormal)
dd_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_lognorm, y_lognorm, z_lognorm, boxsize=L, periodic=periodic)
dr_res_ln = Corrfunc.theory.DD(0, nthreads, r_edges, x_lognorm, y_lognorm, z_lognorm, X2=x_null, Y2=y_null, Z2=z_null, boxsize=L, periodic=periodic)
rr_res_ln = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=L, periodic=periodic)

dd_ln = np.array([x['npairs'] for x in dd_res_ln], dtype=float)
dr_ln = np.array([x['npairs'] for x in dr_res_ln], dtype=float)
rr_ln = np.array([x['npairs'] for x in rr_res_ln], dtype=float)

# calculating landy-szalay
xi_ln = landy_szalay(nd,nr,dd_ln,dr_ln,rr_ln)
xi_dead = landy_szalay(nd,nr,dd_dead,dr_dead,rr_dead)

# add r_avg values to array
ln_xi = np.array([r_avg, xi_ln])
dead_xi = np.array([r_avg, xi_dead])

# save
np.save("lognormal_xi", ln_xi)
np.save("dead_xi", dead_xi)

# loop through the m and b values
for m in m_arr_perL:
    for b in b_arr:
        # define a value in terms of m and b
        a = f"m-{m}-L_b-{b}"
        # load in gradient mock
        grad_mock = np.load(f"gradient_mocks/{grad_dim}D/mocks/grad_mock_{a}.npy")
        x_grad, y_grad, z_grad = grad_mock.T

        # CALCULATING CORRELATION FUNCTION
        # even though len(grad_mock) is not exactly equal to len(lognormal_set)==len(dead_set),
        #   it's close enough that I will approximate nd for grad mock ~ nd for lognormal mock

        # calculating Corrfunc for grad
        dd_res_grad = Corrfunc.theory.DD(1, nthreads, r_edges, x_grad, y_grad, z_grad, boxsize=L, periodic=periodic)
        dr_res_grad = Corrfunc.theory.DD(0, nthreads, r_edges, x_grad, y_grad, z_grad, X2=x_null, Y2=y_null, Z2=z_null, boxsize=L, periodic=periodic)
        rr_res_grad = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=L, periodic=periodic)

        # pull out only pair counts we need (dd,dr,rr) because Corrfunc.theory.DD returns a bunch of other stuff
        dd_grad = np.array([x['npairs'] for x in dd_res_grad], dtype=float)
        dr_grad = np.array([x['npairs'] for x in dr_res_grad], dtype=float)
        rr_grad = np.array([x['npairs'] for x in rr_res_grad], dtype=float)

        # calculating landy-szalay
        xi_grad = landy_szalay(nd,nr,dd_grad,dr_grad,rr_grad)

        # combine correlation function and corresponding r_avg values for saving/plotting later
        grad_xi = np.array([r_avg, xi_grad])

        # save Corrfunc data
        np.save(f"gradient_mocks/{grad_dim}D/xi/grad_xi_{a}_per-{periodic}",grad_xi)

        print(f"m={m}/L, b={b}, done")