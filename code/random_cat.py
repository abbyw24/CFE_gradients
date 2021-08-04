
import os
import time
import numpy as np

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

def generate_random(nbar, boxsize, nx, savepos=None):  # previously had an optional seed argument; do i need to worry about this?
    print("Making random catalog")
    s = time.time()
    nr = float(nbar) * float(boxsize)**3
    random = np.random.uniform(-boxsize/2, boxsize/2, (3,nr))
    print('time: {}'.format(time.time()-s))
    print("Random: {}".format(nr))
    if savepos:
        np.savetxt(savepos, random)     # keeping this saved in txt format since this is what kate had / is in bao_iterative
    return random

# def get_positions(cat):
#     catx = np.array(cat['Position'][:,0]).astype(float)
#     caty = np.array(cat['Position'][:,1]).astype(float)
#     catz = np.array(cat['Position'][:,2]).astype(float)
#     return catx, caty, catz

if __name__=='__main__':
    boxsize_list = [500, 750, 1000, 1500]
    nbar_str_list = ['1e-6', '1e-5', '1e-4', '2e-4', '4e-4']
    for boxsize in boxsize_list:
        for nbar_str in nbar_str_list:
            main(boxsize=boxsize, nbar_str=nbar_str)