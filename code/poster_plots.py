import numpy as np
import matplotlib.pyplot as plt
import os
import globals
globals.initialize_vals()

path_to_data_dir = globals.path_to_data_dir

# load in mock to use
m = 1.0
b = 0.5
lognormal_density = "2e-4"
n_patches = 8
lognorm_file = f"cat_L750_n{lognormal_density}_z057_patchy_lognormal_rlz400"
mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file, m, b)

# mock_info
mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name}.npy"), allow_pickle=True).item()
mock_file_name = mock_info["mock_file_name"]
mock_name = mock_info["mock_name"]
mock_data = mock_info["grad_set"]
L = mock_info["boxsize"]

# patch_info
patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name}.npy"), allow_pickle=True).item()
r_avg = patch_info["r_avg"]
xi_patches = patch_info["xi_patches"]
grad_rec_patches = patch_info["grad_recovered"]

# suave_info
suave_info = np.load(os.path.join(path_to_data_dir, f"suave_data/{lognormal_density}/{mock_file_name}.npy"), allow_pickle=True).item()
grad_rec_suave = suave_info["grad_recovered"]