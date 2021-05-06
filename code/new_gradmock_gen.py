import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_lognormal
import globals

globals.initialize_vals()

grad_dim = globals.grad_dim
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

# we generate L in this script so no need to import

# pick a seed number so that random set stays the same every time (for now)
np.random.seed(123456)

# lognorm file used so far = "lss_data/lognormal_mocks/cat_L750_n3e-4_lognormal_rlz0.bin"

# function to generate gradient mock
def generate_gradmock(grad_dim, path_to_lognorm_file, lognorm_file, output_file, z_max=-50, plot_title=output_file):
    # make sure all inputs have the right form
    assert isinstance(grad_dim, int)
    assert isinstance(path_to_lognorm_file, str)
    assert isinstance(lognorm_file, str)
    assert isinstance(output_file, str)
    assert isinstance(z_max, int or float)
    assert isinstance(plot_title, str)

    # load in lognormal set
    Lx, Ly, Lz, N, data = read_lognormal.read(path_to_lognorm_file+lognorm_file)
        # define boxsize based on mock; and N = number of data points
    L = Lx
        # L = boxsize
    # save boxsize then load in this boxsize for all uses of gradient mock
    np.save("lognormal_data/boxsizes/boxsize_"+lognorm_file, L)

    # save lognormal set
    x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
    xs_clust = np.array([x_lognorm, y_lognorm, z_lognorm])-(L/2)
    np.save("lognormal_data/lognormal_sets/lognormal_set_"+lognorm_file, xs_clust.T)

    # generate a random data set (same size as mock)
    xs_uncl = np.random.uniform(-L/2,L/2,(3,N))
    np.save("lognormal_data/dead_sets/dead_set_"+lognorm_file, xs_uncl.T)

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
    print(f"grad dim = {grad_dim}, w_hat = {w_hat}")

    # for each catalog, make random uniform deviates
    rs_clust = np.random.uniform(size=N)
    rs_uncl = np.random.uniform(size=N)

    # dot product onto the unit vectors
    ws_clust = np.dot(w_hat, xs_clust)
    ws_uncl = np.dot(w_hat, xs_uncl)

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
    xs = np.append(xs_clust.T[I_clust], xs_uncl.T[I_uncl], axis=0)

    # save xs
    np.save(f"gradient_mocks/{grad_dim}D/mocks/mock_data/grad_data_{output_file}", xs)

    # also save clustered and unclustered separately (used to plot in separate colors later)
    np.save(f"gradient_mocks/{grad_dim}D/mocks/mock_data/clust/clust_data_{output_file}", xs_clust_grad)
    np.save(f"gradient_mocks/{grad_dim}D/mocks/mock_data/unclust/unclust_data_{output_file}", xs_uncl_grad)

    # visualisation! (we define z_max cutoff in function parameters)

    # plot all points in same color
    xy_slice = xs[np.where(xs[:,2] < z_max)] # select rows where z < z_max

    fig1 = plt.figure()
    plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice
    plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], color="green", label=w_hat)   # plot vector w_hat (no z)
    plt.axes().set_aspect("equal")      # square aspect ratio
    # plt.ylim((-400,400))
    plt.xlabel("x (Mpc/h)")
    plt.ylabel("y (Mpc/h)")
    plt.title(plot_title)
    plt.legend()
    fig1.savefig(f"gradient_mocks/{grad_dim}D/mocks/plots/gradmock_{output_file}.png")

    # plot different colors for clust and uncl
    fig2 = plt.figure()

    xs_clust_grad = xs_clust.T[I_clust]
    xs_uncl_grad = xs_uncl.T[I_uncl]

    xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
    xy_slice_uncl = xs_uncl_grad[np.where(xs_uncl_grad[:,2] < z_max)]

    plt.plot(xy_slice_clust[:,0], xy_slice_clust[:,1], ',', c="C0", label="clustered")
    plt.plot(xy_slice_uncl[:,0], xy_slice_uncl[:,1], ',', c="orange", label="unclustered")
    plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], c="green", label=w_hat)   # plot vector w_hat (no z)
    plt.axes().set_aspect("equal")      # square aspect ratio
    # plt.ylim((-400,400))
    # plt.xlim((-400,400))
    plt.xlabel("x (Mpc/h)")
    plt.ylabel("y (Mpc/h)")
    plt.title(plot_title)
    plt.legend()
    fig2.savefig(f"gradient_mocks/{grad_dim}D/mocks/plots/color_gradmock_{output_file}.png")

    print(f"gradient generated from {lognorm_file} --> {output_file}")