import numpy as np
import matplotlib.pyplot as plt
import os

def histogram(data1, data2, hist_name, path_to_hist_dir, label1="patches", label2="suave", nbins=10):
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # make sure all inputs have the right form
    assert data1.shape == data2.shape
    assert isinstance(hist_name, str)
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
        plt.hist(data1[:,i], bins=nbins, alpha=0.5, label=label1)
        plt.hist(data2[:,i], bins=nbins, alpha=0.5, label=label2)

        plt.legend()

        fig.savefig(os.path.join(path_to_hist_dir, f"hist_patches_vs_suave_{dim[i]}_{hist_name}.png"))
        plt.cla()