import numpy as np
import matplotlib.pyplot as plt
import os
from center_mock import center_mock
from suave import cf_model
from fourparam_fit import xi_bestfit
import generate_mock_list
import globals

globals.initialize_vals()  # brings in all the default parameters

cat_tag = globals.cat_tag
grad_dim = globals.grad_dim
boxsize = globals.boxsize
lognormal_density = globals.lognormal_density
data_dir = globals.data_dir
grad_dir = globals.grad_dir

nbins = globals.nbins
n_patches = globals.n_patches

mock_file_name_list = generate_mock_list.generate_mock_list()

# instead of the old cf_model (fiducial), we now use the results from the 4-parameter fit (analogous to bao_iterative)
def f_bases(mock_fn, r, x, cov_type = 'identity'):
    assert len(x) == 3

    # load in best-fit parameters
    fit_results = np.load(os.path.join(data_dir, f'bases/4-parameter_fit/{cov_type}/results_gradient_{cat_tag}/basis_gradient_{mock_fn}.npy'), allow_pickle=True).item()
    alpha = fit_results['best_alpha']
    B_sq = fit_results['B_sq']
    a1 = fit_results['a1']
    a2 = fit_results['a2']
    a3 = fit_results['a3']
    C_inv = fit_results['C_inv']

    # xi_mod
    xi_mod = cf_model(alpha*r, cosmo_base=None, redshift=0.57, bias=2.0)
        # how should i be carrying this through to be consistent/limit repetition --> possibility for error?

    # plug best_alpha back in to retrieve the best-fit correlation function
    xi_fit = xi_bestfit(r, xi_mod, B_sq, a1, a2, a3)

    # xi_mod = cf_model(r)  # old fiducial model
    x_pos = np.array([1, x[0], x[1], x[2]])
    
    return xi_mod * x_pos

def patches_lstsq_fit(grad_dim=grad_dim, grad_dir=grad_dir, n_patches=n_patches):

    for i in range(len(mock_file_name_list)):
        mock_info = np.load(os.path.join(grad_dir, f'mock_data/{cat_tag}/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
        mock_fn = mock_info['mock_file_name']
        L = mock_info['boxsize']

        patch_info = np.load(os.path.join(grad_dir, f'patch_data/{cat_tag}/{n_patches}patches/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
        patch_centers = patch_info['patch_centers']

        # center mock around 0
        center_mock(patch_centers, -L/2, L/2)

        r = patch_info['r_avg']
        xi_patches = patch_info['xi_patches']

        # below we "unroll" the array with binned xi in each patch (dimensions nbins x npatches) into a 1D array with length nbins x npatches
        #   in order to perform our least square fit

        # create empty arrays
        X = np.empty((len(r)*len(patch_centers), 4))
        xi = np.empty((len(r)*len(patch_centers), 1))

        # fill the arrays with the relevant data
        for patch in range(len(patch_centers)):
            for n_bin in range(len(r)):
                X_bin = f_bases(mock_fn, r[n_bin], patch_centers[patch])

                xi_bin = xi_patches[patch, n_bin]
                row = patch*len(r) + n_bin
                X[row,:] = X_bin
                xi[row] = xi_bin
        
        C_inv = np.identity(len(xi))

        # performing the fit: xi = X @ theta
        a = X.T @ C_inv @ X
        b = X.T @ C_inv @ xi

        theta, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        # theta = np.linalg.inv(X.T @ C_inv @ X) @ (X.T @ C_inv @ xi)

        b_fit = theta[0]
        m_fit = theta[1:]
    
        # add recovered values to patch info dictionary
        patch_info['theta'] = theta
        patch_info['b_fit'] = b_fit
        patch_info['m_fit'] = m_fit

        # add recovered gradient value to patch info dictionary
        grad_recovered = m_fit/b_fit
        patch_info['grad_recovered'] = grad_recovered

        # change back patch_center values for dictionary saving
        center_mock(patch_centers, 0, L)
        # resave patch info dictionary
        np.save(os.path.join(grad_dir, f'patch_data/{cat_tag}/{n_patches}patches/{mock_fn}'), patch_info, allow_pickle=True)

        print(f"lstsqfit in {n_patches} patches --> {mock_fn}")