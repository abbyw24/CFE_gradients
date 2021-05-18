import numpy as np
import matplotlib.pyplot as plt
import os
import globals

globals.initialize_vals()

m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

n_patches = globals.n_patches

from create_subdirs import create_subdirs

def histogram_exp_vs_rec(patches_data, suave_data, path_to_hist_dir, label1="patches", label2="suave", nbins=10):
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # make sure all inputs have the right form
    assert patches_data.shape == suave_data.shape
    assert isinstance(label1, str)
    assert isinstance(label2, str)
    assert isinstance(path_to_hist_dir, str)

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {dim[i]}")
        plt.xlabel("Recovered Gradient")

        # line at x = 0
        plt.vlines(0, 0, 20, color="black", alpha=0.4)

        # histogram for patches
        plt.hist(patches_data[:,i], bins=nbins, alpha=0.5, label=label1)
        plt.hist(suave_data[:,i], bins=nbins, alpha=0.5, label=label2)

        plt.legend()

        fig.savefig(os.path.join(path_to_hist_dir, f"hist_patches_vs_suave_{dim[i]}.png"))
        plt.cla()