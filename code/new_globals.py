import numpy as np
import os

# every mock should be identifiable via a "mock_file"; I could create an array of absolute mock files and should be able to do everything
#   via that mock_file, and via globals; no internally-defined parameters
# research-summer2020
#   mocks (JUST mock data),  plots (JUST png files),  extracted_vals (expected and recovered vals)
#       giant list of .npy files specified by m, b, and lognorm rlz
#                           patches     suave           patches     suave

def initialize_vals():
    # gradient values
    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock

    global path_to_lognorm_source
    path_to_lognorm_source = "/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy"

    global grad_type
    grad_type = "1rlz_per_m"

    mock_name_list = []

    if grad_type == "1rlz":
        m_arr_perL = np.linspace(-1.0, 1.0, 201)
        b_arr = np.array([0.5])
        lognorm_file = "cat_L750_n2e-4_z057_patchy_lognormal_rlz1"
        for m in m_arr_perL:
            for b in b_arr:
                mock_name = "{lognorm_file}_m-{:.2f}-L_b-{:.2f}".format(m, b)
                mock_name_list.append(mock_name)

    elif grad_type == "1m":
        m = 0.5
        b = 0.5
        lognorm_file_list = []
        for i in range(101):
            lognorm_file_list.append(f"cat_L750_n2e-4_z057_patchy_lognormal_rlz{i}")

        for lognorm_file in lognorm_file_list:
            mock_name = "{lognorm_file}_m-{:.2f}-L_b-{:.2f}".format(m, b)
            mock_name_list.append(mock_name)
    
    elif grad_type == "1rlz_per_m":
        m_arr_perL = np.linspace(-1.0, 1.0, 201)
        b_arr = np.array([0.5])

        lognorm_file_list = []
        for i in range(201):
            lognorm_file_list.append(f"cat_L750_n2e-4_z057_patchy_lognormal_rlz{i}")
        lognorm_file_arr = np.array(lognorm_file_list)

        # make sure each m value corresponds to its own lognorm rlz
        assert len(m_arr_perL) == len(lognorm_file_arr)
        mock_arr = np.vstack((m_arr_perL, lognorm_file_arr))
        print(mock_arr)
    
    else:
        print("'grad_type' must be '1rlz', '1m', or '1rlz_per_m'")
        assert False

    mock_name_arr = np.array(mock_name_list)

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