import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

import read_lognormal
import fetch_lognormal_mocks
import globals
import generate_mock_list
from center_mock import center_mock
globals.initialize_vals()


def is_smooth(tau_sq):
    """Returns True if all values are between 0 and 1, otherwise returns False."""
    bools = (tau_sq >= 0) & (tau_sq <= 1)
    return all(bools)


def generate_gradmocks(L = globals.boxsize,
                        n = globals.lognormal_density,
                        As = globals.As,
                        rlzs = globals.rlzs,
                        data_dir = globals.data_dir,
                        grad_dim = globals.grad_dim,
                        m = globals.m,
                        b = globals.b,
                        input_w = globals.input_w,
                        prints = False, plots = False, z_max = -50, assert_smooth = True):
    """Use global parameters to generate a set of gradient mocks."""
    
    s = time.time()
    
    # generate mock list
    mock_set = generate_mock_list.mock_set(L, n, As=As, data_dir=data_dir, rlzs=rlzs)

    # if an input gradient vector is specified as an argument, set grad_dim accordingly (i.e. ignore globals.grad_dim)
    if input_w:
        grad_dim = len(np.where(np.array(input_w)!=0)[0])
        assert input_w[0] != 0, "first component of input_w must be nonzero"
        if grad_dim > 1:
            assert input_w[1] != 0, "second component of input_w must be nonzero if grad_dim > 1"

    # initialize gradient
    mock_set.add_gradient(grad_dim, m, b)

    ### should this be inside or outside the loop? depends on whether we want w_hat to be the same for all mocks
    # generate unit vector– this is the direction of the gradient
    if grad_dim == 1:
        w_hat = np.array([1.0,0,0])
    elif grad_dim == 2:
        if input_w:
            w_hat = input_w
        else:
            w_hat = np.random.normal(size=3)
        w_hat[2] = 0
    elif grad_dim == 3:
        if input_w:
            w_hat = input_w
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

        ##
        if not prints and i==0:
            print(f'first mock: ', mock_file_name)

        # input gradient
        w = m / (b*L*np.sqrt(3)) * w_hat     # sqrt(3) is included to ensure that threshold^2 is between 0 and 1 for entire mock if m <= 1

        # create dictionary with initial mock info
        mock_dict = {
            'mock_file_name' : mock_file_name,
            'cat_tag' : cat_tag,
            'lognormal_rlz' : ln_file_name,
            'grad_input' : w,
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
        
        # number of data points
        N = ln_dict['N']
        mock_dict['N'] = N

        # boxsize
        L = ln_dict['L']
        mock_dict['L'] = L
        
        # data points
        xs_lognorm = ln_dict['data'].T
        center_mock(xs_lognorm, -L/2, L/2)


        # NULL SET
        # generate a null (unclustered) data set (same size as mock)
        xs_rand = np.random.uniform(-L/2,L/2,(3,N))


        # INJECT GRADIENT
        # for each catalog, make random uniform deviates
        rs_clust = np.random.uniform(size=N)
        rs_uncl = np.random.uniform(size=N)

        # dot product onto the unit vectors
        eta_clust = np.dot(w, xs_lognorm)
        eta_uncl = np.dot(w, xs_rand)

        # threshold
        ts_clust_squared = b*eta_clust + b 
        ts_uncl_squared = b*eta_uncl + b

        # assert that ts range from 0 to 1
        if assert_smooth:
            assert is_smooth(ts_clust_squared)
            assert is_smooth(ts_uncl_squared)

        # desired indices
        I_clust = rs_clust**2 < ts_clust_squared
        I_uncl = rs_uncl**2 > ts_uncl_squared

        # clustered and unclustered sets for gradient
        xs_clust_grad = xs_lognorm.T[I_clust]
        xs_unclust_grad = xs_rand.T[I_uncl]

        # append to create gradient mock data
        xs_grad = np.append(xs_clust_grad, xs_unclust_grad, axis=0)

        # add new data to mock dictionary
        mock_dict['rand_set'] = xs_rand
        mock_dict['clust_set'] = xs_clust_grad
        mock_dict['unclust_set'] = xs_unclust_grad
        mock_dict['data'] = xs_grad

        if i==0:
            print("expected gradient = ", w)
            print("w_hat = ", w_hat)
            print("boxsize = ", L)
            print("galaxy density = ", n)
            print("m = ", m)


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

        if prints:
            print(f"gradient generated --> {grad_dim}D, {mock_file_name}")
    
    print(f"gradient generated --> {grad_dim}D, {cat_tag}, {mock_set.nmocks} mocks")
    total_time = time.time()-s
    print(f"total time = {datetime.timedelta(seconds=total_time)}")