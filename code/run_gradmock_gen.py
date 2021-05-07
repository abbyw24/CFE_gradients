import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
from gradmock_gen import generate_gradmock
import globals

globals.initialize_vals()

grad_dim = globals.grad_dim
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

path_to_lognorm_source = "/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy"           # locally: "lss_data/lognormal_mocks/"
lognorm_file = "cat_L750_n2e-4_z057_patchy_lognormal_rlz0" #.bin
path_to_mocks_dir = f"mocks/{grad_dim}D/{lognorm_file}"

# # loop through m and b arrays
# for m in m_arr_perL:
#     for b in b_arr:
#         mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
#         generate_gradmock(grad_dim, m, b, path_to_lognorm_source, lognorm_file, path_to_mocks_dir, mock_name)

# loop through lognormal realizations with m=0, b=0.5
for n in range(0,101):
    lognorm_file = f"cat_L750_n2e-4_z057_patchy_lognormal_rlz{n}" #.bin
    m = 0.00
    b = 0.50
    mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)

    generate_gradmock(grad_dim, m, b, path_to_lognorm_source, lognorm_file, path_to_mocks_dir, mock_name)