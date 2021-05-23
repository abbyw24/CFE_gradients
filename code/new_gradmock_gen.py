import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import read_lognormal
from create_subdirs import create_subdirs
import new_globals

new_globals.initialize_vals()

path_to_data_dir = new_globals.path_to_data_dir
grad_dim = new_globals.grad_dim
path_to_lognorm_source = new_globals.path_to_lognorm_source
lognorm_file_list = new_globals.lognorm_file_list
m_arr_perL = new_globals.m_arr_perL
b_arr = new_globals.b_arr

mock_name_list = new_globals.mock_name_list

mock_info = new_globals.mock_info

# pick a seed number so that random set stays the same every time (for now)
np.random.seed(123456)

# lognorm file used so far = "lss_data/lognormal_mocks/cat_L750_n3e-4_lognormal_rlz0.bin"

# function to generate gradient mock
def generate_gradmocks(grad_dim=grad_dim, path_to_lognorm_source=path_to_lognorm_source, mock_info=mock_info, z_max=-50):

    # create desired path to mocks directory if it doesn't already exist
    sub_dirs = ["mock_data/rand_sets",
    "mock_data/lognorm_sets",
    "mock_data/clust",
    "mock_data/unclust",
    "mock_data/boxsizes",
    "mock_data/grad_mocks"
    "plots/color_mocks",
    "plots/samecolor_mocks"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    ### should this be inside or outside the loop? depends on whether we want w_hat to be the same for all mocks
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

    for mock_name in mock_info:
        # define mock
        print(mock_name)
        assert False
        mock = mock_info[mock_name]
        lognorm_file = str(mock[0])
        m = float(mock[1])
        b = float(mock[2])

        # LOGNORMAL SET
        Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_lognorm_source, f"{lognorm_file}.bin"))
            # define boxsize based on mock; and N = number of data points
        L = Lx
            # L = boxsize
        # save boxsize then load in this boxsize for all uses of gradient mock
        np.save(os.path.join(path_to_data_dir, f"mock_data/boxsizes/boxsize_{lognorm_file}"), L)

        # save lognormal set to mocks directory
        x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
        xs_lognorm = (np.array([x_lognorm, y_lognorm, z_lognorm])-(L/2))
        np.save(os.path.join(path_to_data_dir, f"mock_data/lognorm_sets/{lognorm_file}"), xs_lognorm)

        # RANDOM SET
        # generate a random data set (same size as mock)
        xs_rand = np.random.uniform(-L/2,L/2,(3,N))
        np.save(os.path.join(path_to_data_dir, f"mock_data/unclust/rand_set_{mock}"), xs_rand)

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

        # assert that ts range from 0 to 1
        # assert np.all(ts_clust > 0)
        # assert np.all(ts_clust < 1)

        # desired indices
        I_clust = rs_clust**2 < ts_clust_squared
        I_uncl = rs_uncl**2 > ts_uncl_squared

        # clustered and unclustered sets for gradient
        xs_clust_grad = xs_lognorm.T[I_clust]
        xs_unclust_grad = xs_rand.T[I_uncl]
        np.save(os.path.join(path_to_data_dir, f"mock_data/clust/clust_{mock}"), xs_clust_grad)
        np.save(os.path.join(path_to_data_dir, f"mock_data/unclust/unclust_{mock}"), xs_unclust_grad)

        # append to create gradient mock data
        xs_grad = np.append(xs_clust_grad, xs_unclust_grad, axis=0)
        np.save(os.path.join(path_to_data_dir, f"mock_data/grad_mocks/{mock}"), xs_grad)

        # visualisation! (we define z_max cutoff in function parameters)

        # plot all points in same color
        xy_slice = xs_grad[np.where(xs_grad[:,2] < z_max)] # select rows where z < z_max

        fig1, ax1 = plt.subplots()
        plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice
        plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], color="green", label=w_hat)   # plot vector w_hat (no z)
        ax1.set_aspect("equal")      # square aspect ratio
        # plt.ylim((-400,400))
        ax1.set_xlabel("x (Mpc/h)")
        ax1.set_ylabel("y (Mpc/h)")
        ax1.set_title(mock)
        ax1.legend()
        fig1.savefig(os.path.join(path_to_data_dir, f"plots/samecolor_mocks/{mock}.png"))
        plt.cla()

        # plot different colors for clust and uncl
        xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
        xy_slice_unclust = xs_unclust_grad[np.where(xs_unclust_grad[:,2] < z_max)]

        fig2, ax2 = plt.subplots()
        plt.plot(xy_slice_clust[:,0], xy_slice_clust[:,1], ',', c="C0", label="clustered")
        plt.plot(xy_slice_unclust[:,0], xy_slice_unclust[:,1], ',', c="orange", label="unclustered")
        plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], c="green", label=w_hat)   # plot vector w_hat (no z)
        ax2.set_aspect("equal")      # square aspect ratio
        # plt.ylim((-400,400))
        # plt.xlim((-400,400))
        ax2.set_xlabel("x (Mpc/h)")
        ax2.set_ylabel("y (Mpc/h)")
        ax2.set_title(mock_name)
        ax2.legend()
        fig2.savefig(os.path.join(path_to_data_dir, f"plots/color_mocks/color_{mock}.png"))
        plt.cla()

        plt.close("all")

        print(f"gradient generated from {lognorm_file} --> {mock}")

generate_gradmocks()