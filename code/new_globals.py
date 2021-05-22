import numpy as np
import os

# every mock should be identifiable via a "mock_file"; I could create an array of absolute mock files and should be able to do everything
#   via that mock_file, and via globals; no internally-defined parameters

def initialize_vals():
    # gradient values
    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock

    global path_to_lognorm_source
    path_to_lognorm_source = "/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy"

    global lognorm_file_list

    global path_to_data_dir
    path_to_data_dir = f"/home/aew492/research-summer2020/{grad_dim}D"

    global grad_type
    grad_type = "1rlz_per_m"

    global n_mocks
    n_mocks = 201

    global mock_info

    global mock_name_list
    mock_name_list = []

    global lognorm_file_list
    global m_arr_perL
    global b_arr

    if grad_type == "1rlz":
        m_arr_perL = np.linspace(-1.0, 1.0, n_mocks)
        b_arr = 0.5 * np.ones([n_mocks])
        lognorm_file_list = ["cat_L750_n2e-4_z057_patchy_lognormal_rlz1"]
        for m in m_arr_perL:
            for b in b_arr:
                mock_name = "{}_m-{:.2f}-L_b-{:.2f}".format(lognorm_file_list[0], m, b)
                mock_name_list.append(mock_name)

    elif grad_type == "1m":
        m = 0.5
        b = 0.5
        m_arr_perL = m * np.ones([n_mocks])
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = []
        for i in range(n_mocks):
            lognorm_file_list.append(f"cat_L750_n2e-4_z057_patchy_lognormal_rlz{i}")

        for lognorm_file in lognorm_file_list:
            mock_name = "{}_m-{:.2f}-L_b-{:.2f}".format(lognorm_file, m, b)
            mock_name_list.append(mock_name)
    
    elif grad_type == "1rlz_per_m":
        b = 0.5
        m_arr_perL = np.linspace(-1.0, 1.0, n_mocks)
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = []
        for i in range(n_mocks):
            lognorm_file_list.append(f"cat_L750_n2e-4_z057_patchy_lognormal_rlz{i}")

        # make sure each m value corresponds to its own lognorm rlz
        assert len(m_arr_perL) == len(lognorm_file_list)

        for i in range(len(m_arr_perL)):
            mock_name = "{}_m-{:.2f}-L_b-{:.2f}".format(lognorm_file_list[i], m_arr_perL[i], b)
            mock_name_list.append(mock_name)
    
    else:
        print("'grad_type' must be '1rlz', '1m', or '1rlz_per_m'")
        assert False
    
    # create dictionary from mock data
    mock_arr = np.array([lognorm_file_list, m_arr_perL, b_arr])
    mock_info = dict(zip(mock_name_list, zip(*mock_arr)))

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

#initialize_vals()