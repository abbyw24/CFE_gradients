import numpy as np

def initialize_vals():
    # gradient values
    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock

    global L
    L = np.load("boxsize.npy")

    global loop
    loop = True         # whether to loop through entire m and b array

    global m_arr_perL
    global b_arr
    # if loop is false, the script will use:
    m = 0.75
    b = 0.5

    if loop == True:
        m_arr_perL = np.linspace(-1.0,1.0,202)
        b_arr = np.array([0.5])
    elif loop == False:
        m_arr_perL = np.array([m])
        b_arr = np.array([b])
    else:
        print("loop must be True or False")
        assert False

    # parameters for landy-szalay:
    #   by default in patchify_xi.xi, periodic=False, rmin=20.0, rmax=100.0, nbins=22
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
    global n_sides
    n_sides = 2         # define number of patches by number of patches per side length

def initialize_path():
    global path_to_dir     # PATH TO RESEARCH DIRECTORY
    # relative path
    path_to_dir = ""
    # if running on cluster
    #path_to_dir = "/home/aew492/research-summer2020/"
    # if running locally
    #path_to_dir = "/Users/abbywilliams/Physics/research-summer2020/"