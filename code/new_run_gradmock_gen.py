import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
from new_gradmock_gen import generate_gradmock
import globals

globals.initialize_vals()

grad_dim = globals.grad_dim
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

path_to_lognorm_file = "lss_data/lognormal_mocks/"
lognorm_file = "cat_L750_n3e-4_lognormal_rlz0.bin"

# loop through m and b arrays
for m in m_arr_perL:
    for b in b_arr:
        output_file = f"m-{m}-L_b-{b}"
        generate_gradmock(1, path_to_lognorm_file, lognorm_file, output_file, plot_title=f"Gradient Mock, m={m}, b={b}")