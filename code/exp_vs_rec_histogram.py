import numpy as np
import matplotlib.pyplot as plt
import os
import globals

globals.initialize_vals()

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

n_patches = globals.n_patches

from create_subdirs import create_subdirs

def histogram(data1, data2, label1, label2, path_to_hist_dir, hist_name, nbins=10):
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # make sure all inputs have the right form
    assert data1.shape == data2.shape
    assert isinstance(hist_name, str)
    assert isinstance(path_to_hist_dir, str)
    assert isinstance(hist_name, str)

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {dim[i]}")
        plt.xlabel("Recovered Gradient")

        # line at x = 0
        plt.vlines(0, 0, 20, color="black", alpha=0.4)

        # histogram for patches
        plt.hist(data1[:,i], bins=nbins, alpha=0.5, label=label1)
        plt.hist(data2[:,i], bins=nbins, alpha=0.5, label=label2)

        plt.legend()

        fig.savefig(os.path.join(path_to_hist_dir, f"{hist_name}_{dim[i]}.png"))
        plt.cla()

def hist_patches_vs_suave_1rlz(mock_name, path_to_mocks_dir):
    # create desired path to directory if it doesn't already exist
    sub_dirs = ["plots/histograms"]
    create_subdirs(path_to_mocks_dir, sub_dirs)

    path_to_hist_dir = os.path.join(path_to_mocks_dir, f"plots/histograms")
    hist_name = f"hist_patches_vs_suave"

    grads_rec_p = []
    grads_rec_s = []
    # loop through m and b values
    for m in m_arr_perL:
        for b in b_arr:
            # mock_name = "m-{:.2f}-L_b-{:.2f}".format(m, b)
            patches_data = np.load(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/exp_vs_rec_vals/patches_exp_vs_rec_{n_patches}patches_{mock_name}.npy"), allow_pickle=True).item()
            grad_rec_p = patches_data["grad_recovered"]
            grads_rec_p.append(grad_rec_p)
            
            suave_data = np.load(os.path.join(path_to_mocks_dir, f"suave/recovered/exp_vs_rec_vals/suave_exp_vs_rec_{mock_name}.npy"), allow_pickle=True).item()
            grad_rec_s = suave_data["grad_recovered"]
            grads_rec_s.append(grad_rec_s)
    grads_rec_p = np.array(grads_rec_p)
    grads_rec_s = np.array(grads_rec_s)

    histogram(grads_rec_p, grads_rec_s, "patches", "suave", path_to_hist_dir, hist_name)