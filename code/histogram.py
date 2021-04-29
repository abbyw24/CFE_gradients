import numpy as np
import matplotlib.pyplot as plt
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
L = globals.L
loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_sides = globals.n_sides

n_patches = n_sides**3

dim = { 1:"y", 2:"z" }
method = ["suave", "patches"]

def patches(method):
    if method == "patches":
        return f"_{n_patches}patches"
    else:
        return ""

# loop through first suave and then patches
for i in dim:
    # create plot
    fig = plt.figure()
    plt.title(f"Histogram of Recovered Gradient, {dim[i]}")
    plt.xlabel("Recovered Gradient")

    # line at x = 0
    plt.vlines(0, 0, 1, color="black", alpha=0.5)

    for j in method:
        # create an array of recovered gradient values for y and z
        grads_recovered = []

        for m in m_arr_perL:
            for b in b_arr:
                data = np.load(f"gradient_mocks/{grad_dim}D/{j}/exp_vs_rec_vals/{j}_exp_vs_rec_vals_m-{m}-L_b-{b}{patches(j)}.npy", allow_pickle=True).item()
                grads_recovered.append(data["grad_recovered"])

        grads_recovered = np.array(grads_recovered)

        # histogram array
        plt.hist(grads_recovered[:,i], bins=10, alpha=0.5, label=j)

    plt.legend()
    plt.show()
    
    fig.savefig(f"gradient_mocks/{grad_dim}D/exp_vs_rec_hist_{dim[i]}.png")