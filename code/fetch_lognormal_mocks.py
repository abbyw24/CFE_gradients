import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

import read_lognormal
import globals
globals.initialize_vals()

cat_tag = globals.cat_tag
nmocks = globals.nmocks
data_dir = globals.data_dir

# for a single mock type (boxsize, number density, amplitude), fetch all the mocks from Kate's files,
#   and resave them to my own catalogs directory in .npy format
def fetch_ln_mocks(cat_tag, mock_dir='/scratch/ksf293/mocks/lognormal'):
    s = time.time()
    path_to_mocks_dir = os.path.join(mock_dir, f'cat_{cat_tag}')

    for rlz in range(nmocks):
        lognorm_fn = f'cat_{cat_tag}_lognormal_rlz{rlz}'
        new_fn = f'{cat_tag}_rlz{rlz}_lognormal'

        # read in mock from kate's files (.bin format)
        Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_mocks_dir, f'{lognorm_fn}.bin'))
        L = Lx      # boxsize
        x, y, z, vx, vy, vz = data.T
        mock_data = np.array([x, y, z]).T
        
        # resave mock data as .npy
        new_dir = os.path.join(data_dir, f'catalogs/lognormal/{cat_tag}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        save_fn = os.path.join(new_dir, f'{new_fn}.npy')
        np.save(save_fn, {'data' : mock_data, 'L' : L, 'N' : N})
        print(f"{lognorm_fn}.bin in /ksf293 --> {new_fn}.npy in /catalogs")
    total_time = time.time() - s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")