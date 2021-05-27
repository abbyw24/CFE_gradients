import numpy as np
import matplotlib.pyplot as plt
import os
from create_subdirs import create_subdirs
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
lognormal_density = globals.lognormal_density
path_to_data_dir = globals.path_to_data_dir
mock_file_name_list = globals.mock_file_name_list
mock_name_list = globals.mock_name_list

n_patches = globals.n_patches

# ** this is currently for the x dimension only **

# define function to import recovered fit values and nicely compare expected to recovered gradient in plot
def patches_exp_vs_rec(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches, z_max=-50, scale=200):
    # make sure all inputs have the right form
    assert isinstance(path_to_data_dir, str)
    for x in [grad_dim, n_patches, z_max, scale]:
        assert isinstance(x, int)

    # create the needed subdirectories
    sub_dirs = [
        f"plots/patches/{lognormal_density}/{n_patches}patches/exp_vs_rec"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)
    
    for i in range(len(mock_file_name_list)):
        # load in mock and patch info
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        mock_file_name = mock_info["mock_file_name"]
        mock_name = mock_info["mock_name"]
        grad_expected = mock_info["grad_expected"]
        L = mock_info["boxsize"]

        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        grad_recovered = patch_info["grad_recovered"]
        ratio_rec_exp = patch_info["ratio_rec_exp"]

        # expected gradient (only in x direction)
        print("expected gradient (m_input/b_input)w_hat =", grad_expected)

        # mean squared error just to see for now how close we are
        mean_sq_err = (1/len(grad_expected))*np.sum((grad_recovered - grad_expected)**2)
        print(f"mean squared error = {mean_sq_err}")

        # add mean squared error to patches dictionary
        patch_info["mean_sq_err"] = mean_sq_err

        # projection of recovered onto expected
        grad_exp_norm = np.linalg.norm(grad_expected)
        print(grad_exp_norm)
        print(grad_exp_norm**2)
        assert False

        # proj_rec_onto_exp = (np.dot(grad_recovered,grad_expected)/grad_exp_norm**2)*grad_expected

        # residual
        residual = grad_recovered - grad_expected

        # load in mock data
        xs_clust_grad = mock_info["clust_set"]
        xs_unclust_grad = mock_info["unclust_set"]

        xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
        xy_slice_unclust = xs_unclust_grad[np.where(xs_unclust_grad[:,2] < z_max)]

        fig, ax = plt.subplots()

        plt.plot(xy_slice_clust[:,0], xy_slice_clust[:,1], ',', c="C0")
        plt.plot(xy_slice_unclust[:,0], xy_slice_unclust[:,1], ',', c="orange")

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

        fig.savefig(os.path.join(path_to_data_dir, f"plots/patches/{lognormal_density}/{n_patches}patches/exp_vs_rec/{mock_file_name}.png"))
        plt.cla()
        plt.close("all")

        # resave patch info dictionary
        np.save(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name}"), patch_info, allow_pickle=True)
    
    patches_exp_vs_rec()