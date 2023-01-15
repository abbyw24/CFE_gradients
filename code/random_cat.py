import os
import time
import numpy as np
import Corrfunc

import globals
globals.initialize_vals()

def main(boxsize=globals.boxsize, nbar_str=globals.lognormal_density, data_dir=globals.data_dir, nx=globals.randmult):
    """Generate a random catalog given global parameters, and calculate the RR term, if it does not already exist."""

    s = time.time()

    tag = '_L{}_n{}'.format(boxsize, nbar_str)

    cat_dir = os.path.join(data_dir, 'catalogs/randoms')
    if not os.path.isdir(cat_dir):
        os.makedirs(cat_dir)
    
    # making random catalogs
    print("Making random catalogs for {}".format(tag))

    rand_fn = '{}/rand{}_{}x.dat'.format(cat_dir, tag, nx)
    nbar = float(nbar_str)
    boxsize = float(boxsize)

    if not os.path.isfile(rand_fn):
        random = generate_random(nbar, boxsize, nx, savepos=rand_fn)
    else:
        print("File already exists! Exiting.", rand_fn)
    
    # calculating rr term
    print("Calculating rr term for {}".format(tag))

    rr_save_dir = os.path.join(globals.data_dir, f'catalogs/randoms/rr_terms')
    if not os.path.exists(rr_save_dir):
        os.makedirs(rr_save_dir)

    rr_fn = '{}/rr_res_rand{}_{}x.npy'.format(rr_save_dir, tag, nx)

    if not os.path.isfile(rr_fn):   
        get_rr_terms(random, savepos=rr_fn)
    else:
        print("File already exists! Exiting.", rr_fn)
    
    print('time: {}'.format(time.time()-s))

def generate_random(nbar, boxsize, nx, savepos=None):
    """Generate a random catalog given a number density, boxsize, and multiplier."""
    nr = nx * float(nbar) * int(boxsize)**3
    random = np.random.uniform(0, boxsize, (int(nr),3))
    if savepos:
        np.savetxt(savepos, random)
    return random

def get_rr_terms(rand_set, savepos=None, nthreads=globals.nthreads, rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, periodic=False):
    """Calculate the RR term for a given random catalog."""
    print("Calculating rr term")
    assert len(rand_set.T) == 3, 'check input random shape'
    x_rand, y_rand, z_rand = rand_set.T
    r_edges = np.linspace(rmin, rmax, nbins+1)
    rr_res = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=periodic)
    if savepos:
        np.save(savepos, rr_res, allow_pickle=True)

# if __name__=='__main__':
#     main(boxsize=750, nbar_str='2e-5', nx=3)