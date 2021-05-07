import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
import os

# we generate L in this script so no need to import

# pick a seed number so that random set stays the same every time (for now)
np.random.seed(123456)

# lognorm file used so far = "lss_data/lognormal_mocks/cat_L750_n3e-4_lognormal_rlz0.bin"

# function to generate gradient mock
def generate_gradmock(grad_dim, m, b, path_to_lognorm_source, lognorm_file, path_to_mocks_dir, mock_name, z_max=-50):
    # make sure all inputs have the right form
    for x in [path_to_lognorm_source, lognorm_file, path_to_mocks_dir, mock_name]:
        assert isinstance(x, str)
    for x in [grad_dim, m, b, z_max]:
        assert isinstance(x, (int, float))

    # create desired path to mocks directory if it doesn't already exist
    for sub_dir in ["grad_mocks", "clust", "unclust", "plots"]:
        if not os.path.exists(f"{path_to_mocks_dir}/{sub_dir}"):
            os.makedirs(f"{path_to_mocks_dir}/{sub_dir}")
            print(f"created path {path_to_mocks_dir}/{sub_dir}")

    # load in lognormal set
    Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_lognorm_source, f"{lognorm_file}.bin"))
        # define boxsize based on mock; and N = number of data points
    L = Lx
        # L = boxsize
    # save boxsize then load in this boxsize for all uses of gradient mock
    np.save(os.path.join(path_to_mocks_dir, f"boxsize"), L)

    # save lognormal set to mocks directory
    x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
    xs_clust = (np.array([x_lognorm, y_lognorm, z_lognorm])-(L/2))
    np.save(os.path.join(path_to_mocks_dir, "lognormal_set"), xs_clust)

    # generate a random data set (same size as mock)
    xs_unclust = np.random.uniform(-L/2,L/2,(3,N))
    np.save(os.path.join(path_to_mocks_dir, f"dead_set"), xs_unclust)

    # generate unit vector– this is the direction of the gradient
    if grad_dim == 1:
        w_hat = np.array([1.0,0,0])
    elif grad_dim == 2:
        w_hat = np.random.normal(size=3)
        w_hat[2] = 0
    elif grad_dim == 3:
        w_hat = np.random.normal(size=3)
    else:
        print("Invalid dimension; must be 1, 2, or 3")
        assert False

    # normalize w_hat and print out result
    w_hat /= np.linalg.norm(w_hat)
    ### might want to come back and save w_hat later

    # print("grad dim = {}, w_hat = {}, m = {:.2f}, b = {:.2f}".format(grad_dim, w_hat, m, b))

    # for each catalog, make random uniform deviates
    rs_clust = np.random.uniform(size=N)
    rs_uncl = np.random.uniform(size=N)

    # dot product onto the unit vectors
    ws_clust = np.dot(w_hat, xs_clust)
    ws_uncl = np.dot(w_hat, xs_unclust)

    # threshold
    ts_clust_squared = (m/L) * ws_clust + b  
    ts_uncl_squared = (m/L) * ws_uncl + b

    # assert that ts range from 0 to 1
    # assert np.all(ts_clust > 0)
    # assert np.all(ts_clust < 1)

    # desired indices
    I_clust = rs_clust**2 < ts_clust_squared
    I_uncl = rs_uncl**2 > ts_uncl_squared

    # append
    xs = np.append(xs_clust.T[I_clust], xs_unclust.T[I_uncl], axis=0)

    # save xs
    np.save(os.path.join(path_to_mocks_dir, f"grad_mocks/gradmock_data_{mock_name}"), xs)

    # also save clustered and unclustered separately (used to plot in separate colors later)
    xs_clust_grad = xs_clust.T[I_clust]
    xs_unclust_grad = xs_unclust.T[I_uncl]

    np.save(os.path.join(path_to_mocks_dir, f"clust/clust_data_{mock_name}"), xs_clust_grad)
    np.save(os.path.join(path_to_mocks_dir, f"unclust/unclust_data_{mock_name}"), xs_unclust_grad)

    # visualisation! (we define z_max cutoff in function parameters)

    # plot all points in same color
    xy_slice = xs[np.where(xs[:,2] < z_max)] # select rows where z < z_max

    fig1, ax1 = plt.subplots()
    plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice
    plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], color="green", label=w_hat)   # plot vector w_hat (no z)
    ax1.set_aspect("equal")      # square aspect ratio
    # plt.ylim((-400,400))
    ax1.set_xlabel("x (Mpc/h)")
    ax1.set_ylabel("y (Mpc/h)")
    ax1.set_title(mock_name)
    ax1.legend()
    fig1.savefig(os.path.join(path_to_mocks_dir, f"plots/mock_{mock_name}.png"))
    plt.cla()

    # plot different colors for clust and uncl
    xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
    xy_slice_uncl = xs_unclust_grad[np.where(xs_unclust_grad[:,2] < z_max)]

    fig2, ax2 = plt.subplots()
    plt.plot(xy_slice_clust[:,0], xy_slice_clust[:,1], ',', c="C0", label="clustered")
    plt.plot(xy_slice_uncl[:,0], xy_slice_uncl[:,1], ',', c="orange", label="unclustered")
    plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], c="green", label=w_hat)   # plot vector w_hat (no z)
    ax2.set_aspect("equal")      # square aspect ratio
    # plt.ylim((-400,400))
    # plt.xlim((-400,400))
    ax2.set_xlabel("x (Mpc/h)")
    ax2.set_ylabel("y (Mpc/h)")
    ax2.set_title(mock_name)
    ax2.legend()
    fig2.savefig(os.path.join(path_to_mocks_dir, f"plots/color_mock_{mock_name}.png"))
    plt.cla()

    plt.close("all")

    print(f"gradient generated from {lognorm_file} --> {mock_name}")