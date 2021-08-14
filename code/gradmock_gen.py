import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from numpy.random import lognormal
import read_lognormal
from create_subdirs import create_subdirs
import globals
import generate_mock_list
globals.initialize_vals()

path_to_data_dir = globals.path_to_data_dir
grad_dim = globals.grad_dim
boxsize = globals.boxsize
grad_type = globals.grad_type
lognormal_density = globals.lognormal_density

mock_vals = generate_mock_list.generate_mock_list(extra=True)
path_to_lognorm_source = mock_vals['path_to_lognorm_source']
mock_file_name_list = mock_vals['mock_file_name_list']
mock_name_list = mock_vals['mock_name_list']
lognorm_file_list = mock_vals['lognorm_file_list']
m_arr = mock_vals["m_arr"]
b_arr = mock_vals["b_arr"]

# # pick a seed number so that random set stays the same every time (for now)
# np.random.seed(123456)

# function to generate gradient mock
def generate_gradmocks(grad_type=grad_type, grad_dim=grad_dim, path_to_lognorm_source=path_to_lognorm_source,
                        mock_file_name_list=mock_file_name_list, plots=False, z_max=-50):
    
    # create desired path to mocks directory if it doesn't already exist
    tag = f'L{int(boxsize)}_n{lognormal_density}'
    mock_dir = f'mock_data/{tag}'
    colormock_dir = f'plots/color_mocks/{tag}'
    samecolormock_dir = f'plots/samecolor_mocks/{tag}'

    sub_dirs = [
        mock_dir,
        colormock_dir,
        samecolormock_dir
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
        assert False, 'Invalid dimension; must be 1, 2, or 3'

    # normalize w_hat
    w_hat /= np.linalg.norm(w_hat)

    for i in range(len(mock_file_name_list)):
        # create dictionary with mock info
        mock_info = {
            "mock_file_name" : mock_file_name_list[i],
            'mock_name' : mock_name_list[i],
            "lognorm_rlz" : lognorm_file_list[i],
            "w_hat" : w_hat,
            "m" : m_arr[i],
            "b" : b_arr[i],
            "lognorm_density" : lognormal_density
        }

        # redefine dictionary values for simplicity
        mock_file_name = str(mock_info["mock_file_name"])
        mock_name = str(mock_info["mock_name"])
        lognorm_file = str(mock_info["lognorm_rlz"])
        m = float(mock_info["m"])
        b = float(mock_info["b"])


        # LOGNORMAL SET
        Lx, Ly, Lz, N, data = read_lognormal.read(os.path.join(path_to_lognorm_source, f"{lognorm_file}.bin"))
            # define boxsize based on mock; and N = number of data points
        # make sure the boxsize from data equals the boxsize we specified in the file name
        assert float(Lx) == float(boxsize), 'boxsize from data does not match boxsize from globals.py'

        # boxsize
        L = Lx
        mock_info["boxsize"] = L

        # expected gradient
        mock_info["grad_expected"] = m/(b*L)*w_hat

        # save lognormal set to mocks directory
        x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
            # data is initially loaded in from 0 to L; we want to shift down by L/2 to center around 0
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

        # assert that ts range from 0 to 1
        # assert np.all(ts_clust > 0)
        # assert np.all(ts_clust < 1)

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


        # VISUALIZATION
        if plots == True:
            # plot all points in same color
            xy_slice = xs_grad[np.where(xs_grad[:,2] < z_max)] # select rows where z < z_max

            fig1, ax1 = plt.subplots()
            plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice
            # plot vector w_hat (no z)
            a = 0.35*L     # controls width of w_hat vector in plot
            plt.arrow(-a, 0, 2*a, 0, color="black", lw=2, head_width = .1*a, head_length=.2*a, length_includes_head=True, zorder=100, label=w_hat)
            ax1.set_aspect("equal")      # square aspect ratio
            ax1.set_xlabel("x (Mpc/h)")
            ax1.set_ylabel("y (Mpc/h)")
            if grad_type == "1mock":
                ax1.set_title("")
            else:
                ax1.set_title(mock_name)
            ax1.legend()
            fig1.savefig(os.path.join(path_to_data_dir, f'{samecolormock_dir}/{lognormal_density}/{mock_file_name}.png'))
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
            fig2.savefig(os.path.join(path_to_data_dir, f'{colormock_dir}/{lognormal_density}/color_{mock_file_name}.png'))
            plt.cla()

            plt.close("all") 

        # save dictionary
        mock_dict_fn = os.path.join(path_to_data_dir, f'{mock_dir}/{mock_file_name_list[i]}')
        np.save(mock_dict_fn, mock_info)

        print(f"gradient generated --> {mock_file_name}")