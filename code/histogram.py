import numpy as np
import matplotlib.pyplot as plt

def histogram(hist_type, grad_dim, data1, data2, label1="patches", label2="suave", nbins=10):
    dim = {
            0 : "x",
            1 : "y",
            2 : "z"
            }

    # check that recovered gradient array is in the right form
    assert data1.shape == data2.shape
    # check that dim is dictionary
    assert isinstance(dim, dict)

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

        fig.savefig(f"gradient_mocks/{grad_dim}D/exp_vs_rec_hist_{dim[i]}_{hist_type}.png")
        plt.cla()