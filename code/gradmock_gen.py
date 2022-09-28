import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

import read_lognormal
import fetch_lognormal_mocks
import globals
import generate_mock_list
globals.initialize_vals()


def generate_gradmocks(nmocks = globals.nmocks,
                        L = globals.boxsize,
                        n = globals.lognormal_density,
                        As = globals.As,
                        rlzs = globals.rlzs,
                        data_dir = globals.data_dir,
                        grad_dim = globals.grad_dim,
                        m = globals.m,
                        b = globals.b,
                        input_w_hat = None,
                        plots = False, z_max = -50):
    """Use global variables to generate a set of gradient mocks."""
    
    s = time.time()
    
    # generate mock list
    mock_set = generate_mock_list.mock_set(nmocks, L, n, As=As, data_dir=data_dir, rlzs=rlzs)

    # initialize gradient
    mock_set.add_gradient(grad_dim, m, b)

    # mock_vals = generate_mock_list.generate_mock_list(cat_tag=cat_tag, extra=True)
    # path_to_lognorm_source = mock_vals['path_to_lognorm_source']
    # mock_file_name_list = mock_vals['mock_file_name_list']
    # lognorm_file_list = mock_vals['lognorm_file_list']
    # m_arr = mock_vals['m_arr']
    # b_arr = mock_vals['b_arr']
    
    # create desired path to mocks directory if it doesn't already exist
    if not os.path.exists(mock_set.grad_dir):
        os.makedirs(mock_set.grad_dir)
        print(f"created path {mock_set.grad_dir}")


    ### should this be inside or outside the loop? depends on whether we want w_hat to be the same for all mocks
    # generate unit vector– this is the direction of the gradient
    if grad_dim == 1:
        w_hat = np.array([1.0,0,0])
    elif grad_dim == 2:
        if input_w_hat:
            w_hat = input_w_hat
        else:
            w_hat = np.random.normal(size=3)
        w_hat[2] = 0
    elif grad_dim == 3:
        if input_w_hat:
            w_hat = input_w_hat
        else:
            w_hat = np.random.normal(size=3)
    else:
        assert False, "Invalid dimension; must be 1, 2, or 3"

    # normalize w_hat
    w_hat /= np.linalg.norm(w_hat)

    # loop through each lognormal realization and inject the specified gradient into each one to create a new mock
    for i, rlz in enumerate(mock_set.rlzs):

        # unpack realization-specific variables
        cat_tag = mock_set.cat_tag
        mock_file_name = mock_set.mock_fn_list[i]
        ln_file_name = mock_set.ln_fn_list[i]
        m = mock_set.m_arr[i]
        b = mock_set.b_arr[i]

        # create dictionary with initial mock info
        mock_dict = {
            'mock_file_name' : mock_file_name,
            'cat_tag' : cat_tag,
            'lognormal_rlz' : ln_file_name,
            'w_hat' : w_hat,
            'm' : m,
            'b' : b,
        }

        # LOGNORMAL SET
        # try fetching the desired lognormal catalogs from my own directories, otherwise fetch from Kate's
        try:
            ln_dict = np.load(os.path.join(data_dir, f'catalogs/lognormal/{cat_tag}/{ln_file_name}.npy'), allow_pickle=True).item()
        except FileNotFoundError:
            print("Fetching lognormal catalogs...")
            fetch_lognormal_mocks.fetch_ln_mocks(cat_tag, mock_set.rlzs)
            ln_dict = np.load(os.path.join(data_dir, f'catalogs/lognormal/{cat_tag}/{ln_file_name}.npy'), allow_pickle=True).item()
        
        # data points
        xs_lognorm = ln_dict['data'].T
        
        # number of data points
        N = ln_dict['N']

        # boxsize
        L = ln_dict['L']

        # expected gradient
        grad_expected = m/(b*L)*w_hat

        # # save lognormal set to mocks directory
        # x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
        #     # data is initially loaded in from 0 to L; we want to shift down by L/2 to center around 0
        # xs_lognorm = (np.array([x_lognorm, y_lognorm, z_lognorm])-(L/2))
        # velocities = np.array([vx_lognorm, vy_lognorm, vz_lognorm])
        # mock_info['lognorm_set'] = xs_lognorm
        # mock_info['velocities'] = velocities


        # NULL SET
        # generate a null (unclustered) data set (same size as mock)
        xs_rand = np.random.uniform(-L/2,L/2,(3,N))


        # INJECT GRADIENT
        # for each catalog, make random uniform deviates
        rs_clust = np.random.uniform(size=N)
        rs_uncl = np.random.uniform(size=N)

        # dot product onto the unit vectors
        eta_clust = np.dot(w_hat, xs_lognorm)
        eta_uncl = np.dot(w_hat, xs_rand)

        # threshold
        ts_clust_squared = (m/L) * eta_clust + b 
        ts_uncl_squared = (m/L) * eta_uncl + b

        # assert that ts range from 0 to 1
        # assert np.all(ts_clust > 0)
        # assert np.all(ts_clust < 1)

        # desired indices
        I_clust = rs_clust**2 < ts_clust_squared
        I_uncl = rs_uncl**2 > ts_uncl_squared

        # clustered and unclustered sets for gradient
        xs_clust_grad = xs_lognorm.T[I_clust]
        xs_unclust_grad = xs_rand.T[I_uncl]

        # append to create gradient mock data
        xs_grad = np.append(xs_clust_grad, xs_unclust_grad, axis=0)

        # add new data to mock dictionary
        mock_dict['grad_expected'] = grad_expected
        mock_dict['rand_set'] = xs_rand
        mock_dict['clust_set'] = xs_clust_grad
        mock_dict['unclust_set'] = xs_unclust_grad
        mock_dict['grad_set'] = xs_grad

        # save our gradient mock dictionary to catalogs directory
        cat_dir = os.path.join(data_dir, f'catalogs/gradient/{grad_dim}D/{cat_tag}')
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
        cat_fn = os.path.join(cat_dir, mock_file_name)

        np.save(cat_fn, mock_dict)


        # VISUALIZATION
        if plots == True:

            # create directories, if they don't already exist
            colormock_dir = f'plots/color_mocks/{cat_tag}'
            samecolormock_dir = f'plots/samecolor_mocks/{cat_tag}'

            sub_dirs = [
                colormock_dir,
                samecolormock_dir
            ]
            create_subdirs(grad_dir, sub_dirs)

            # plot all points in same color
            xy_slice = xs_grad[np.where(xs_grad[:,2] < z_max)] # select rows where z < z_max

            fig1, ax1 = plt.subplots()
            plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice
            # plot vector w_hat (no z)
            a = 0.35*L     # controls width of w_hat vector in plot
            plt.arrow(-a, 0, 2*a, 0, color='black', lw=2, head_width = .1*a, head_length=.2*a, length_includes_head=True, zorder=100, label=w_hat)
            ax1.set_aspect('equal')      # square aspect ratio
            ax1.set_xlabel("x (Mpc/h)")
            ax1.set_ylabel("y (Mpc/h)")
            if mock_type == '1mock':
                ax1.set_title("")
            else:
                ax1.set_title(mock_file_name)
            ax1.legend()
            fig1.savefig(os.path.join(grad_dir, f'{samecolormock_dir}/{mock_file_name}.png'))
            plt.cla()

            # plot different colors for clust and uncl
            xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
            xy_slice_unclust = xs_unclust_grad[np.where(xs_unclust_grad[:,2] < z_max)]

            fig2, ax2 = plt.subplots()
            plt.plot(xy_slice_clust[:,0], xy_slice_clust[:,1], ',', c='C0', label="clustered")
            plt.plot(xy_slice_unclust[:,0], xy_slice_unclust[:,1], ',', c='orange', label="unclustered")
            plt.plot(xy_slice[:,0], (w_hat[1]/w_hat[0])*xy_slice[:,0], c='green', label=w_hat)   # plot vector w_hat (no z)
            ax2.set_aspect('equal')      # square aspect ratio
            # plt.ylim((-400,400))
            # plt.xlim((-400,400))
            ax2.set_xlabel("x (Mpc/h)")
            ax2.set_ylabel("y (Mpc/h)")
            ax2.set_title(mock_file_name)
            ax2.legend()
            fig2.savefig(os.path.join(grad_dir, f'{colormock_dir}/color_{mock_file_name}.png'))
            plt.cla()

            plt.close('all') 

        # # save dictionary
        # mock_dict_fn = os.path.join(grad_dir, f'{mock_dir}/{mock_file_name_list[i]}')
        # np.save(mock_dict_fn, mock_info)

        print(f"gradient generated --> {grad_dim}D, {mock_file_name}")
    
    total_time = time.time()-s
    print(datetime.timedelta(seconds=total_time))