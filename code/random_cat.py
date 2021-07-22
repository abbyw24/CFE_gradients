
import os
import time
import numpy as np

import nbodykit
from nbodykit.lab import *

import globals
globals.initialize_vals()

def main(boxsize=globals.boxsize, nbar_str=globals.lognormal_density, nx=globals.As):

    tag = '_L{}_n{}'.format(boxsize, nbar_str)
    
    print("Making random catalogs for {}".format(tag))

    cat_dir = 'research-summer2020_output/catalogs/randoms'
    if not os.path.isdir(cat_dir):
        os.makedirs(cat_dir)

    rand_fn = '{}/rand{}_{}x.dat'.format(cat_dir, tag, nx)
    nbar = float(nbar_str)
    boxsize = float(boxsize)

    if not os.path.isfile(rand_fn):
        random = generate_random(nbar, boxsize, nx, savepos=rand_fn)
    else:
        print("File already exists! Exiting.", rand_fn)

def generate_random(nbar, boxsize, nx, seed=41, savepos=None):
    print("Making random catalog")
    s = time.time()
    random = nbodykit.source.catalog.uniform.UniformCatalog(nx*nbar, boxsize, seed=seed)
    print('time: {}'.format(time.time()-s))
    nr = random.csize
    print("Random: {}".format(nr))
    if savepos:
        randompos = get_positions(random)
        np.savetxt(savepos, np.array(randompos).T)
    return random

def get_positions(cat):
    catx = np.array(cat['Position'][:,0]).astype(float)
    caty = np.array(cat['Position'][:,1]).astype(float)
    catz = np.array(cat['Position'][:,2]).astype(float)
    return catx, caty, catz

if __name__=='__main__':
    main()