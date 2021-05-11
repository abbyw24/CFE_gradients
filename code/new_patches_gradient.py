import numpy as np
import matplotlib.pyplot as plt
import os
from create_subdirs import create_subdirs
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim

n_patches = globals.n_patches

# ** this is currently for the x dimension only **

# define function to import recovered fit values and nicely compare expected to recovered gradient in plot
def patches_exp_vs_rec_vals(grad_dim, m, b, path_to_mocks_dir, mock_name, z_max=-50, scale=200):


    # create the needed subdirectories
    create_subdirs(f"{path_to_mocks_dir}/patches", ["plots/exp_vs_rec", "lst_sq_fit/exp_vs_rec_vals"])

    # load in recovered gradient
    recovered_vals = np.load(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/recovered_vals_{n_patches}patches_{mock_name}.npy"), allow_pickle=True).item()
    b_fit = recovered_vals["b_fit"]
    m_fit_x = recovered_vals["m_fit_x"]
    m_fit_y = recovered_vals["m_fit_y"]
    m_fit_z = recovered_vals["m_fit_z"]

    grad_recovered = (1/b_fit)*np.array([m_fit_x,m_fit_y,m_fit_z])
    print("recovered gradient (m_fit/b_fit) =", grad_recovered)

    # check that there is a corresponding boxsize file
    boxsize_file = os.path.join(path_to_mocks_dir, f"boxsize.npy")
    assert os.path.exists(boxsize_file)
    L = np.load(boxsize_file)

    # expected gradient (only in x direction)
    grad_expected = np.array([m/(b*L),0,0])
    print("expected gradient (m_input/b_input)w_hat =", grad_expected)

    # mean squared error just to see for now how close we are
    mean_sq_err = (1/len(grad_expected))*np.sum((grad_recovered - grad_expected)**2)
    print(f"mean squared error = {mean_sq_err}")

    # projection of recovered onto expected
    grad_exp_norm = np.linalg.norm(grad_expected)
    proj_rec_onto_exp = (np.dot(grad_recovered,grad_expected)/grad_exp_norm**2)*grad_expected

    # residual
    residual = grad_recovered - grad_expected

    # load in mock data
    xs_clust_grad = np.load(os.path.join(path_to_mocks_dir, f"clust/clust_data_{mock_name}.npy"))
    xs_unclust_grad = np.load(os.path.join(path_to_mocks_dir, f"unclust/unclust_data_{mock_name}.npy"))

    xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
    xy_slice_uncl = xs_unclust_grad[np.where(xs_unclust_grad[:,2] < z_max)]

    fig, ax = plt.subplots()

    plt.scatter(xy_slice_clust[:,0], xy_slice_clust[:,1], marker=',', c="C0")
    plt.scatter(xy_slice_uncl[:,0], xy_slice_uncl[:,1], marker=',', c="orange")

    # plot expected, recovered, and projection from origin (only in xy)
    V = np.array([grad_expected, grad_recovered, proj_rec_onto_exp])
    colors = np.array(["red", "black", "blue"])
    labels = np.array([r"Expected", r"Recovered", r"Proj. of $\vec{r}$ onto $\vec{e}$"])
    plot_array = np.column_stack((V, colors, labels))
    a = scale*L     # scale factor

    for i in range(len(V)-1):
        plt.plot([0,a*V[i,0]], [0,a*V[i,1]], label=plot_array[i,4], color=plot_array[i,3], alpha=0.8, linewidth=2)
    # plot residual as triangle
    plt.plot([a*grad_expected[0],a*grad_recovered[0]], [a*grad_expected[1],a*residual[1]], color="black", alpha=0.6, label="Residual", linewidth=2)

    ax.set_title(f"Expected vs. Recovered Clustering Gradient: \n {mock_name}, {n_patches} patches")
    ax.set_aspect("equal")      # square aspect ratio
    ax.set_xlabel("x (Mpc/h)")
    ax.set_ylabel("y (Mpc/h)")
    plt.ylim((-400,400))
    plt.xlim((-400,400))
    plt.legend()

    fig.savefig(os.path.join(path_to_mocks_dir, f"patches/plots/exp_vs_rec/patches_exp_vs_rec_{n_patches}patches_{mock_name}.png"))
    plt.cla()

    # save recovered and expected values to dictionary
    exp_vs_rec_vals = {
        "m" : m,
        "b" : b,
        "n_patches" : n_patches,
        "grad_expected" : grad_expected,
        "grad_recovered" : grad_recovered,
        "mean_sq_err" : mean_sq_err
    }

    np.save(os.path.join(path_to_mocks_dir, f"patches/lst_sq_fit/exp_vs_rec_vals/patches_exp_vs_rec_{n_patches}patches_{mock_name}"), exp_vs_rec_vals, allow_pickle=True)

    print(" ")      # line break for nice loop print formatting