import numpy as np
import matplotlib.pyplot as plt

def histogram(grads_recovered_patches, grads_recovered_suave, dim, hist_type):
    # check that recovered gradient array is in the right form
    assert grads_recovered_patches.shape == grads_recovered_suave.shape

    # best way to assert that the inputs are the proper types?

    # loop through desired dimensions with patches and suave
    for i in dim:
        # create plot
        fig = plt.figure()
        plt.title(f"Histogram of Recovered Gradient, {dim[i]}")
        plt.xlabel("Recovered Gradient")

        # line at x = 0
        plt.vlines(0, 0, 20, color="black", alpha=0.4)

        # histogram for patches
        plt.hist(grads_recovered_patches[:,i], bins=10, alpha=0.5, label="patches")
        plt.hist(grads_recovered_suave[:,i], bins=10, alpha=0.5, label="suave")

        plt.legend()
        plt.show()

        fig.savefig(f"gradient_mocks/{grad_dim}D/exp_vs_rec_hist_{dim[i]}_{hist_type}.png")