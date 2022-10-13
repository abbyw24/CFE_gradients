def initialize_vals():

    # BASIC PARAMETERS (non-gradient)

    global nmocks
    nmocks = 401

    global boxsize
    boxsize = 500

    global lognormal_density
    lognormal_density = "1e-4"

    global As
    As = 2

    global rlzs     # if we want to use specific realization numbers, instead of range(nmocks)
    rlzs = None

    global data_dir
    data_dir = f'/scratch/aew492/CFE_gradients_output'


    # MOCK TYPE

    global mock_type
    mock_type = 'gradient'


    # GRADIENT PARAMETERS

    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock

    global m
    m = 0.1

    global b
    b = 0.5


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

    global n_patches
    n_patches = 8 
