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
lognorm_file = f"cat_L750_n{lognormal_density}_z057_patchy_lognormal_rlz400"
mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file, m, b)

# mock_info
mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name}.npy"), allow_pickle=True).item()
mock_file_name = mock_info["mock_file_name"]
mock_name = mock_info["mock_name"]
mock_data = mock_info["grad_set"]
L = mock_info["boxsize"]

# patch_info
# suave_info