import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
from center_mock import center_mock
from suave import cf_model
import generate_mock_list
import globals

globals.initialize_vals()  # brings in all the default parameters

def xi_bestfit(r, xi_mod, B_sq, a1, a2, a3):

    xi_bestfit = B_sq*xi_mod + a1/r**2 + a2/r + a3

    return xi_bestfit

# instead of the old cf_model (fiducial), we now use the results from the 4-parameter fit (analogous to bao_iterative)
def f_bases(mock_fn, r, x, cov_type = 'scipy', cat_tag=globals.cat_tag, data_dir=globals.data_dir, grad_dim=globals.grad_dim):
    assert len(x) == 3

    # load in best-fit parameters
    fit_results = np.load(os.path.join(data_dir, f'bases/{grad_dim}D/4-parameter_fit/{cov_type}/results_gradient_{cat_tag}/basis_gradient_{mock_fn}.npy'), allow_pickle=True).item()
    alpha = fit_results['best_alpha']
    B_sq = fit_results['B_sq']
    a1 = fit_results['a1']
    a2 = fit_results['a2']
    a3 = fit_results['a3']
    cov = fit_results['cov']

    # xi_mod
    xi_mod = cf_model(alpha*r, cosmo_base=None, redshift=0.57, bias=2.0)

    # plug best_alpha back in to retrieve the best-fit correlation function
    xi_fit = xi_bestfit(r, xi_mod, B_sq, a1, a2, a3)

    x_pos = np.array([1, x[0], x[1], x[2]])
    
    return xi_fit * x_pos, cov

def patches_lstsq_fit(cat_tag=globals.cat_tag, grad_dim=globals.grad_dim, grad_dir=globals.grad_dir,
                        npatches=globals.n_patches, mock_range=None, test=False, cov_type='diag'):
    
    s = time.time()

    mock_file_name_list = generate_mock_list.generate_mock_list(cat_tag=cat_tag)

    mock_range = mock_range if mock_range else range(len(mock_file_name_list))

    for i in mock_range:
        mock_info = np.load(os.path.join(grad_dir, f'mock_data/{cat_tag}/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
        mock_fn = mock_info['mock_file_name']
        L = mock_info['boxsize']

        patch_info = np.load(os.path.join(grad_dir, f'patch_data/{cat_tag}/{npatches}patches/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
        patch_centers = patch_info['patch_centers']
        # patch_centers.shape == (npatches, 3)

        # center mock around 0
        center_mock(patch_centers, -L/2, L/2)

        r = patch_info['r_avg']
        nbins = len(r)
        xi_patches = patch_info['xi_patches']
        # xi_patches.shape == (npatches, nbins)

        # below we "unroll" the array with binned xi in each patch (dimensions nbins x npatches) into a 1D array with length nbins x npatches
        #   in order to perform our least square fit

        # create empty arrays
        X = np.empty((nbins*npatches, 4))
        xi = np.empty((nbins*npatches, 1))

        # fill the arrays with the relevant data
        for patch in range(npatches):
            for nbin in range(nbins):
                X_bin, cov = f_bases(mock_fn, r[nbin], patch_centers[patch], cat_tag=cat_tag)

                xi_bin = xi_patches[patch, nbin]
                row = patch*nbins + nbin
                X[row,:] = X_bin
                xi[row] = xi_bin
    
        # "tile" covariance array (since we now have nbins x npatches)
        cov_tiled = np.tile(cov, (npatches,npatches))

        if cov_type == 'diag':
            cov_big = np.zeros(cov_tiled.shape)
            np.fill_diagonal(cov_big, np.diag(cov_tiled))
        elif cov_type == 'full':
            cov_big = cov_tiled
        else:
            assert False, "cov_type must be 'diag' or 'full'"

        # performing the fit: xi = X @ theta

        a = X.T @ np.linalg.solve(cov_big, X)
        b = X.T @ np.linalg.solve(cov_big, xi)

        # a theta = b
        theta, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        # theta = np.linalg.inv(X.T @ C_inv @ X) @ (X.T @ C_inv @ xi)

        # b_fit = theta[0]
        # m_fit = theta[1:]
    
        # add recovered values to patch info dictionary
        patch_info['theta'] = theta
        # patch_info['b_fit'] = b_fit
        # patch_info['m_fit'] = m_fit
        #     # should i be saving b_fit and m_fit, if this info is contained in theta ?

        # add recovered gradient value to patch info dictionary
        grad_recovered = theta[1:]/theta[0]
        patch_info['grad_recovered'] = grad_recovered

        # change back patch_center values for dictionary saving
        center_mock(patch_centers, 0, L)
        # resave patch info dictionary
        if test:
            test_dir = os.path.join(grad_dir, f'patch_data/{cat_tag}/{npatches}patches/test_dir')
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            np.save(os.path.join(test_dir, mock_fn), patch_info, allow_pickle=True)
        else:
            np.save(os.path.join(grad_dir, f'patch_data/{cat_tag}/{npatches}patches/{mock_fn}'), patch_info, allow_pickle=True)

        print(f"lstsqfit in {npatches} patches --> {mock_fn}")
    
    total_time = time.time()-s
    print(f"fit to patches --> {cat_tag}, {len(mock_range)} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")