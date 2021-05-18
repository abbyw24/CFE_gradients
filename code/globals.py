import numpy as np

def initialize_vals():
    # gradient values
    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock

    global path_to_lognorm_source
    path_to_lognorm_source = "/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy"

    global lognorm_file_arr
    # if loop is false, the script will use:
    lognorm_file = "cat_L750_n2e-4_z057_patchy_lognormal_rlz1" #.bin        # which lognormal realization to use

    global m_arr_perL
    global b_arr
    # if loop is false, the script will use:
    m = 0.5
    b = 0.5

    global grad_type
    grad_type = "1m"

    if grad_type == "1rlz":
        m_arr_perL = np.linspace(-1.0, 1.0, 201)
        b_arr = np.array([0.5])

        lognorm_file_arr = np.array([lognorm_file])

    elif grad_type == "1m":
        m_arr_perL = np.array([m])
        b_arr = np.array([b])

        lognorm_file_list = []
        for i in range(101):
            lognorm_file_list.append(f"cat_L750_n2e-4_z057_patchy_lognormal_rlz{i}")
        lognorm_file_arr = np.array(lognorm_file_list)
    
    else:
        print("'grad_type' must be '1rlz' or '1m'")
        assert False

    # parameters for landy-szalay:
    #   by default in patchify_xi.xi, periodic=False, rmin=20.0, rmax=100.0, nbins=22
    global randmult
    randmult = 2
    global periodic
    periodic = False
    global rmin
    rmin = 20.0
    global rmax
    rmax = 100.0
    global nbins
    nbins = 22
    global nthreads
    nthreads = 12

    # patch values
    global n_patches
    n_patches = 8         # define number of patches by number of patches per side length