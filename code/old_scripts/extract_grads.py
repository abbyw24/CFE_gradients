import numpy as np
import os

import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim

lognorm_file_arr = globals.lognorm_file_arr

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

n_patches = globals.n_patches

def extract_grads_exp_vs_rec():
    grads_exp_p = []
    grads_rec_p = []
    grads_exp_s = []
    grads_rec_s = []
    # loop through lognormal array / m and b values
    for lognorm_file in lognorm_file_arr:
        for m in m_arr_perL:
            for b in b_arr:
                mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
                path_to_mocks_dir = f"mocks/{grad_dim}D/{lognorm_file}"
                
                patches_data = np.load(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/exp_vs_rec_vals/patches_exp_vs_rec_{n_patches}patches_{mock_name}.npy"), allow_pickle=True).item()
                grad_exp_p = patches_data["grad_expected"]
                grad_rec_p = patches_data["grad_recovered"]
                grads_exp_p.append(grad_exp_p)
                grads_rec_p.append(grad_rec_p)
                
                suave_data = np.load(os.path.join(path_to_mocks_dir, f"suave/recovered/exp_vs_rec_vals/suave_exp_vs_rec_{mock_name}.npy"), allow_pickle=True).item()
                grad_exp_s = suave_data["grad_expected"]
                grad_rec_s = suave_data["grad_recovered"]
                grads_exp_s.append(grad_exp_s)
                grads_rec_s.append(grad_rec_s)
    grads_exp_p = np.array(grads_exp_p)
    grads_rec_p = np.array(grads_rec_p)
    grads_exp_s = np.array(grads_exp_s)
    grads_rec_s = np.array(grads_rec_s)

    grads = {
        "grads_exp_p" : grads_exp_p,
        "grads_rec_p" : grads_rec_p,
        "grads_exp_s" : grads_exp_s,
        "grads_rec_s" : grads_rec_s,
    }

    return grads