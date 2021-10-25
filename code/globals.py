import numpy as np
import os

from create_subdirs import create_subdirs

# every mock should be identifiable via a "mock_file"; I could create an array of absolute mock files and should be able to do everything
#   via that mock_file, and via globals; no internally-defined parameters

def initialize_vals():
    
    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock

    global data_dir
    data_dir = f'/scratch/aew492/research-summer2020_output'

    global grad_dir
    grad_dir = os.path.join(data_dir, f'gradients/{grad_dim}D')

    global boxsize
    boxsize = 750

    global lognormal_density
    lognormal_density = "1e-4"

    global As
    As = 2

    global cat_tag
    if As == 1:
        cat_tag = f'L{boxsize}_n{lognormal_density}_z057_patchy'
    else:
        cat_tag = f'L{boxsize}_n{lognormal_density}_z057_patchy_As{As}x'

    global mock_type
    mock_type = '1m'     # choose from '1rlz', '1m', '1rlz_per_m', '1mock', or 'lognormal'

    global m
    m = 1.0

    global b
    b = 0.5

    global rlz
    rlz = 0

    global n_mocks
    n_mocks = 401 

    global mocks_info

    global lognorm_file_list
    global m_arr
    global b_arr

    # parameters for cf estimators:
    global randmult
    randmult = 3 
    global periodic
    periodic = False
    global rmin
    rmin = 20.0
    global rmax
    rmax = 140.0
    global nbins
    nbins = 22
    global nthreads
    nthreads = 12

    # patch values
    global n_patches
    n_patches = 8          # should be n_sides**3

#initialize_vals()
