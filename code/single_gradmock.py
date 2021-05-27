import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from numpy.random import lognormal
import read_lognormal
from create_subdirs import create_subdirs

def gen_single_gradmock(grad_dim, m, b, lognorm_file, output_dir, mock_file_name, mock_name, z_max=-50):
    # create dictionary for mock info
    mock_info = {
        "grad_dim" : grad_dim,
        "mock_name" : mock_name,
        "m" : m,
        "b" : b
    }

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
    
    # normalize w_hat
    w_hat /= np.linalg.norm(w_hat)
    mock_info["w_hat"] = w_hat

    # LOGNORMAL SET
    Lx, Ly, Lz, N, data = read_lognormal.read(lognorm_file)
        # define boxsize based on mock; and N = number of data points
    L = Lx
        # L = boxsize
    mock_info["boxsize"] = L
    
    # save lognormal set to mocks directory
    x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
    xs_lognorm = (np.array([x_lognorm, y_lognorm, z_lognorm])-(L/2))
    velocities = np.array([vx_lognorm, vy_lognorm, vz_lognorm])
    mock_info["lognorm_set"] = xs_lognorm
    mock_info["velocities"] = velocities

    # RANDOM SET
    # generate a random data set (same size as mock)
    xs_rand = np.random.uniform(-L/2,L/2,(3,N))
    mock_info["rand_set"] = xs_rand

    # INJECT GRADIENT
    # for each catalog, make random uniform deviates
    rs_clust = np.random.uniform(size=N)
    rs_uncl = np.random.uniform(size=N)

    # dot product onto the unit vectors
    ws_clust = np.dot(w_hat, xs_lognorm)
    ws_uncl = np.dot(w_hat, xs_rand)

    # threshold
    ts_clust_squared = (m/L) * ws_clust + b 
    ts_uncl_squared = (m/L) * ws_uncl + b

    # desired indices
    I_clust = rs_clust**2 < ts_clust_squared
    I_uncl = rs_uncl**2 > ts_uncl_squared

    # clustered and unclustered sets for gradient
    xs_clust_grad = xs_lognorm.T[I_clust]
    xs_unclust_grad = xs_rand.T[I_uncl]
    mock_info["clust_set"] = xs_clust_grad
    mock_info["unclust_set"] = xs_unclust_grad

    # append to create gradient mock data
    xs_grad = np.append(xs_clust_grad, xs_unclust_grad, axis=0)
    mock_info["grad_set"] = xs_grad

    # plot all points in same color
    xy_slice = xs_grad[np.where(xs_grad[:,2] < z_max)] # select rows where z < z_max

    fig1, ax1 = plt.subplots()
    plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice
    plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], color="green", label=w_hat)   # plot vector w_hat (no z)
    ax1.set_aspect("equal")      # square aspect ratio
    # plt.ylim((-400,400))
    ax1.set_xlabel("x (Mpc/h)")
    ax1.set_ylabel("y (Mpc/h)")
    ax1.set_title(mock_name)
    ax1.legend()
    fig1.savefig(os.path.join(output_dir, f"{mock_file_name}.png"))
    plt.cla()

    # save mock info
    np.save(os.path.join(output_dir, mock_file_name), mock_info, allow_pickle=True)

grad_dim = 1
m = 1.0
b = 0.5
lognormal_density = "2e-4"

lognorm_name = "cat_L750_n2e-4_z057_patchy_lognormal_rlz1"
lognorm_file = f"/scratch/ksf293/mocks/lognormal/cat_L750_n2e-4_z057_patchy/{lognorm_name}.bin"

create_subdirs("/scratch/aew492", [f"other_outputs/{lognorm_name}"])
output_dir = "/scratch/aew492/other_outputs"
mock_file_name = "{}_m-{}-L_b-{}".format(lognorm_name, m, b)
mock_name = "n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m, b)

gen_single_gradmock(grad_dim, m, b, lognorm_file, output_dir, mock_file_name, mock_name)