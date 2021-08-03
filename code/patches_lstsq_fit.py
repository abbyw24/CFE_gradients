import numpy as np
import matplotlib.pyplot as plt
import os
from create_subdirs import create_subdirs
from patchify_xi import center_mock
from suave import cf_model
import generate_mock_list
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
lognormal_density = globals.lognormal_density
path_to_data_dir = globals.path_to_data_dir

nbins = globals.nbins
n_patches = globals.n_patches

mock_file_name_list = generate_mock_list.generate_mock_list()

def f_bases(r, x):
    assert len(x) == 3
    xi_mod = cf_model(r)
    x_pos = np.array([1, x[0], x[1], x[2]])
    
    return xi_mod * x_pos

def patches_lstsq_fit(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, n_patches=n_patches):
    # create the needed subdirectories
    sub_dirs = [
        f"plots/patches/{lognormal_density}/{n_patches}patches/lst_sq_fit/new_fit"
    ]
    create_subdirs(path_to_data_dir, sub_dirs)

    for i in range(len(mock_file_name_list)):
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        mock_file_name = mock_info["mock_file_name"]
        L = mock_info["boxsize"]

        patch_info = np.load(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        patch_centers = patch_info["patch_centers"]

        # center mock around 0
        center_mock(patch_centers, -L/2, L/2)

        r = patch_info["r_avg"]
        xi_patches = patch_info["xi_patches"]

        # below we "unroll" the array with binned xi in each patch (dimensions nbins x npatches) into a 1D array with length nbins x npatches
        #   in order to perform our least square fit

        # create empty arrays
        X = np.empty((len(r)*len(patch_centers), 4))
        xi = np.empty((len(r)*len(patch_centers), 1))

        # fill the arrays with the relevant data
        for patch in range(len(patch_centers)):
            for n_bin in range(len(r)):
                X_bin = f_bases(r[n_bin], patch_centers[patch])
                xi_bin = xi_patches[patch, n_bin]
                row = patch*len(r) + n_bin
                X[row,:] = X_bin
                xi[row] = xi_bin
        
        C = np.identity(len(xi))
        C_inv = np.linalg.inv(C)

        # least square fit!
            # xi = X @ theta
        theta = np.linalg.inv(X.T @ C_inv @ X) @ (X.T @ C_inv @ xi)

        b_fit = theta[0]
        m_fit = theta[1:]
    
        # add recovered values to patch info dictionary
        patch_info["theta"] = theta
        patch_info["b_fit"] = b_fit
        patch_info["m_fit"] = m_fit

        # add recovered gradient value to patch info dictionary
        grad_recovered = m_fit/b_fit
        patch_info["grad_recovered"] = grad_recovered

        # change back patch_center values for dictionary saving
        center_mock(patch_centers, 0, L)
        # resave patch info dictionary
        np.save(os.path.join(path_to_data_dir, f"patch_data/{lognormal_density}/{n_patches}patches/{mock_file_name}"), patch_info, allow_pickle=True)

        print(f"lstsqfit in {n_patches} patches --> {mock_file_name}")