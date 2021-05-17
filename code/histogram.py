import numpy as np
import matplotlib.pyplot as plt
import os

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

        fig.savefig(os.path.join(path_to_hist_dir, hist_name))
        plt.cla()

def hist_patches_vs_suave(patches_data, suave_data, path_to_mocks_dir):
    # create desired path to mocks directory if it doesn't already exist
    sub_dirs = ["plots/histograms"]
    create_subdirs(path_to_mocks_dir, sub_dirs)

    path_to_hist_dir = os.path.join(path_to_mocks_dir, f"plots/histograms")
    hist_name = f"hist_patches_vs_suave_{dim[i]}.png"

    histogram(patches_data, suave_data, label1="patches", label2="suave", path_to_hist_dir, hist_name)