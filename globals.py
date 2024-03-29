import numpy as np

def initialize_vals():

    # BASIC PARAMETERS (non-gradient)

    global rlzs
    rlzs = 1000

    global boxsize
    boxsize = 1000

    global lognormal_density
    lognormal_density = "2e-4"
    
    global As
    As = 2

    global data_dir
    data_dir = f'/scratch/aew492/CFE_gradients_output'


    # MOCK TYPE

    global mock_type
    mock_type = 'gradient'


    # GRADIENT PARAMETERS

    global grad_dim
    grad_dim = 3        # dimension of w_hat in gradient mock

    global m
    m = 0.667

    global b
    b = 0.5

    global input_w
    input_w = None #[0.8660254, 0.5, 0.]

    global same_dir
    same_dir = False
    if input_w:
        same_dir = True


    # CF ESTIMATOR PARAMETERS

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
    global ncont
    ncont = 2000
    global nthreads
    nthreads = 12


    # PATCH PARAMETERS

    global npatches
    npatches = 8 
