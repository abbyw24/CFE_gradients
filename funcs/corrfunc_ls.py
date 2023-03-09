import numpy as np
import Corrfunc


def compute_ls(data, rand_set, periodic, nthreads, rmin, rmax, nbins, rr_fn=None, prints=False):
    """Run the Landy-Szalay estimator using Corrfunc"""

    # parameters
    r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    nr = len(rand_set)

    x, y, z = data.T
    x_rand, y_rand, z_rand = rand_set.T

    dd_res = Corrfunc.theory.DD(1, nthreads, r_edges, x, y, z, periodic=periodic, output_ravg=True)
    if prints:
        print("DD calculated")
    dr_res = Corrfunc.theory.DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, periodic=periodic)
    if prints:
        print("DR calculated")
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=periodic)
    if prints:
        print("RR calculated")

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd,nd,nr,nr,dd,dr,dr,rr)
    if prints:
        print("3d counts converted to cf")

    return r_avg, results_xi