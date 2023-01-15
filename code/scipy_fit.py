import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import datetime
import scipy

from suave import cf_model

import generate_mock_list

import globals
globals.initialize_vals()

"""
INPUTS:
- mock galaxy catalog
- Landy-Szalay results

RETURNS:
- a set of basis functions (analogous to bao_iterative)
"""


def chisq(x, A, y, cov):
    resid = y - A @ x
    return resid @ np.linalg.solve(cov, resid)


def jacobian(x, A, y, cov):
    resid = y - A@x
    return -2*A.T @ np.linalg.solve(cov, resid)


def scipy_fit(alpha, xi_ls, r_avg, cov,
                cosmo_base=None, redshift=0.57, bias=2.0):
    """Fit a xi function to Landy-Szalay results given an alpha value and covariances."""

    nbins = len(r_avg)

    # xi_mod
    xi_mod = cf_model(alpha*r_avg, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    # feature matrix
    A = np.array([xi_mod, 1/r_avg**2, 1/r_avg, np.ones(nbins)]).T

    # initial parameters guess
    x0 = np.ones(4)

    # bounds for parameters (B^2 > 0)
    bnds = ((0, None), (None, None), (None, None), (None, None))
    
    minimized_results = scipy.optimize.minimize(chisq, x0, args=(A,xi_ls,cov), method='L-BFGS-B', jac=jacobian, bounds=bnds)
    M = minimized_results['x']

    # plug M (best-fit params) into xi equation
    xi_fit = A @ M

    return xi_fit, M


def find_best_alpha(xi_ls, r_avg, cov,
                    alpha_min=0.75, alpha_max=1.25, nalphas=501):
    """Compute the alpha value which minimizes chi-squared, given binned 2pcf results and covariances."""

    nbins = len(r_avg)
    alpha_grid = np.linspace(alpha_min, alpha_max, nalphas)

    Ms = np.empty((nalphas, 4))     # design matrix
    xi_fits = np.empty((nalphas, nbins))

    for i in range(nalphas):
        xi_fit, M = scipy_fit(alpha_grid[i], xi_ls, r_avg, cov)
        Ms[i] = M
        xi_fits[i] = xi_fit
    
    # chi-squared test: find the alpha which minimizes chi-squared
    chi_squareds = np.ones(nalphas)
    for i in range(nalphas):
        diff = xi_ls - xi_fits[i]
        chi_squared = np.dot(diff, np.linalg.solve(cov, diff))
        chi_squareds[i] = chi_squared
    
    min_chi_squared = min(chi_squareds)
    best_alpha = alpha_grid[chi_squareds.argmin()]
    M = Ms[chi_squareds.argmin()]

    return best_alpha, alpha_grid, min_chi_squared, M


def fourparam_fit(r_avg, xi_ls, cov):
    """Compute a 4-parameter fit to binned 2pcf results for a single galaxy catalog; return results in a dictionary."""
    
    # find the best alpha (the one that minimizes chi-squared)
    best_alpha, _, min_chi_squared, M = find_best_alpha(xi_ls, r_avg, cov)

    # unpack M matrix = best-fit parameters
    B_sq, a1, a2, a3 = M.T

    results_dict = {
        'best_alpha' : best_alpha,
        'chi_squared' : min_chi_squared,
        'B_sq' : B_sq,
        'a1' : a1,
        'a2' : a2,
        'a3' : a3,
        'cov' : cov
    }

    return results_dict    


def main(mock_type=globals.mock_type,
            L=globals.boxsize, n=globals.lognormal_density, As=globals.As, rlzs=globals.rlzs,
            grad_dim=globals.grad_dim, m=globals.m, b=globals.b,
            nbins=globals.nbins, randmult=globals.randmult, data_dir=globals.data_dir):
    
    s = time.time()

    # generate the mock set parameters
    mock_set = generate_mock_list.MockSet(L, n, As=As, data_dir=data_dir, rlzs=rlzs)
    cat_tag = mock_set.cat_tag

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    # save directory
    save_dir = os.path.join(data_dir, f'bases/scipy/{mock_set.mock_path}/{cat_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # calculate covariance matrix from the Landy-Szalay results
    xi_lss = np.empty((mock_set.nmocks, nbins))
    for i, mock_fn in enumerate(mock_set.mock_fn_list):
        xi_results = np.load(os.path.join(data_dir, f'{mock_set.mock_path}/ls/{cat_tag}/xi_ls_{randmult}x_{mock_fn}.npy'), allow_pickle=True)
        r_avg = xi_results[0]
        xi_lss[i] = xi_results[1]
    cov = np.cov(xi_lss.T)          # cov.shape==(nbins,nbins)

    for i, mock_fn in enumerate(mock_set.mock_fn_list):

        # load in data
        # r_avg is the same for all mocks
        xi_ls = xi_lss[i]

        # find the best alpha (the one that minimizes chi-squared)
        best_alpha, _, min_chi_squared, M = find_best_alpha(xi_ls, r_avg, cov)

        # save best-fit cf
        save_dir = os.path.join(data_dir, f'bases/4-parameter_fit/scipy/{mock_set.mock_path}/results_{cat_tag}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # unpacking M matrix
        B_sq, a1, a2, a3 = M.T
        
        save_fn = os.path.join(save_dir, f'basis_{mock_fn}')
        save_data = {
            'best_alpha' : best_alpha,
            'chi_squared' : min_chi_squared,
            'B_sq' : B_sq,
            'a1' : a1,
            'a2' : a2,
            'a3' : a3,
            'cov' : cov
        }
        np.save(save_fn, save_data)

        # here we only save the resulting best-fit values (as opposed to the resulting bestfit correlation function) in order to
        #   reduce redundancy and increase flexibility– B, a1, a2, and a3 can be used with any r_avg to output xi
    
    total_time = time.time()-s
    print(f"scipy_fit --> {save_dir}, {mock_set.nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


if __name__=='__main__':
    main()