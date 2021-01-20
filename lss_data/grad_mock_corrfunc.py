import numpy as np
import matplotlib.pyplot as plt
import math
import Corrfunc

# define which gradient we're working with
    # choices are m = (0, 0.3, 1, 10) and b = (0, 0.5, 0.75, 1)
m_arr = np.array([0.0,0.3,1.0,10.0]) #/L
b_arr = np.array([0.0,0.5,0.75,1.0])

# parameters for Corrfunc, used in loop
nthreads = 1
periodic = False

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

# loop through the possible m and b values
for m in m_arr:
    for b in b_arr:
        # define a value in terms of m and b
        a = "m-"+str(m)+"-L_b-"+str(b)
        # load in lognormal, dead, and gradient mocks
        grad_mock = np.load("gradient_mocks/grad_mock_"+a+".npy")
        # define N and boxsize, L
        N = len(grad_mock)
        L = 2.0 * math.ceil(max(grad_mock[:,0]))

        # CALCULATING CORRFUNC
        # define NULL (random) set for Corrfunc
        nr = 2*N
        null_set = np.random.uniform(-L/2, L/2, (nr,3))

        # relabeling mocks / pulling out x, y, z values
        x_grad = grad_mock[:,0]
        y_grad = grad_mock[:,1]
        z_grad = grad_mock[:,2]

        x_null = null_set[:,0]
        y_null = null_set[:,1]
        z_null = null_set[:,2]

        # parameter
        nd = N

        # calculating Corrfunc for grad
        dd_res_grad = Corrfunc.theory.DD(1, nthreads, r_edges, x_grad, y_grad, z_grad, boxsize=L, periodic=periodic)
        dr_res_grad = Corrfunc.theory.DD(0, nthreads, r_edges, x_grad, y_grad, z_grad, X2=x_null, Y2=y_null, Z2=z_null, boxsize=L, periodic=periodic)
        rr_res_grad = Corrfunc.theory.DD(1, nthreads, r_edges, x_null, y_null, z_null, boxsize=L, periodic=periodic)

        # pull out only pair counts we need (dd,dr,rr) because Corrfunc.theory.DD returns a bunch of other stuff
        dd_grad = np.array([x['npairs'] for x in dd_res_grad], dtype=float)
        dr_grad = np.array([x['npairs'] for x in dr_res_grad], dtype=float)
        rr_grad = np.array([x['npairs'] for x in rr_res_grad], dtype=float)

        # calculating landy-szalay for grad, ln and dead
        xi_grad = landy_szalay(nd,nr,dd_grad,dr_grad,rr_grad)

        # combine corr funcs for grad, lognormal, and dead into one array for saving
        grad_xi = np.array([r_avg,xi_grad])

        # save Corrfunc data
        np.save("gradient_mocks/grad_xi_"+a+"_per-"+str(periodic),grad_xi)

        print("m="+str(m)+", b="+str(b)+", done!")