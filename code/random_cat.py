import os
import time
import numpy as np
import Corrfunc

import globals
globals.initialize_vals()

def main(boxsize=globals.boxsize, nbar_str=globals.lognormal_density, nx=3):

    tag = '_L{}_n{}'.format(boxsize, nbar_str)

    cat_dir = '/scratch/aew492/research-summer2020_output/catalogs/randoms'
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

def generate_random(nbar, boxsize, nx, savepos=None):
    print("Making random catalog")
    s = time.time()
    nr = nx * float(nbar) * int(boxsize)**3
    random = np.random.uniform(0, boxsize, (int(nr),3))
    print('time: {}'.format(time.time()-s))
    print("N: {}".format(int(nr)))
    if savepos:
        np.savetxt(savepos, random)
    return random

def get_rr_terms(rand_set, savepos=None, nthreads=globals.nthreads, rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, periodic=False):
    print("Calculating rr term")
    s = time.time()
    assert len(rand_set.T) == 3, 'check input random shape'
    x_rand, y_rand, z_rand = rand_set.T
    r_edges = np.linspace(rmin, rmax, nbins+1)
    rr_res = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=periodic)
    print('time: {}'.format(time.time()-s))
    if savepos:
        np.save(savepos, rr_res, allow_pickle=True)

if __name__=='__main__':
    boxsizes = [500, 750, 1000, 1500]
    densities = ['1e-6', '1e-5', '1e-4', '2e-4', '4e-4', '1e-3']
    for L in boxsizes:
        for n in densities:
            main(boxsize=L, nbar_str=n, nx=3)