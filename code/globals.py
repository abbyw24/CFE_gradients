def initialize_patchvals():
    global grad_dim
    grad_dim = 1        # dimension of w_hat in gradient mock (just for file saving purposes here;
                        #   doesn't affect actual computation)
    global n_sides
    n_sides = 2         # define number of patches by number of patches per side length
    global loop
    loop = True         # whether to loop through entire m and b array
    # if loop is false, the script will use:
    global m
    m = 0.75
    global b
    b = 0.5
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

def initialize_path():
    global path_to_dir     # PATH TO RESEARCH DIRECTORY
    path_to_dir = ""
    # if running on cluster
    #path_to_dir = "/home/aew492/research-summer2020"
    # if running locally
    #path_to_dir = "/Users/abbywilliams/Physics/research"