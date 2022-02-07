import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

from suave import cf_model

import generate_mock_list

import globals
globals.initialize_vals()

def xi_bestfit(r, xi_mod, B_sq, a1, a2, a3):

    xi_bestfit = B_sq*xi_mod + a1/r**2 + a2/r + a3

    return xi_bestfit

def fit_to_cf_model(alpha, xi_ls, r_avg, C_inv, cosmo_base=None, redshift=0.57, bias=2.0):

    nbins = len(r_avg)

    # xi_mod
    xi_mod = cf_model(alpha*r_avg, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    # matrices
    Y = xi_ls       # results of L-S estimator on entire mock (i.e. no patches); vector of length nbins

    A = np.array([xi_mod, 1/r_avg**2, 1/r_avg, np.ones(nbins)]).T       # feature matrix

    # performing the fit
    a = A.T @ C_inv @ A
    b = A.T @ C_inv @ Y

    M, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    B_sq, a1, a2, a3 = M

    xi_fit = xi_bestfit(r_avg, xi_mod, B_sq, a1, a2, a3)

    return xi_fit, M

# for the standard approach, we perform this fit for several different alpha values and find the one which minimizes chi-squared
def find_best_alpha(xi_ls, r_avg, C_inv, alpha_min=0.75, alpha_max=1.25, nalphas=51):

    nbins = len(r_avg)
    alpha_grid = np.linspace(alpha_min, alpha_max, nalphas)

    Ms = np.empty((nalphas, 4))     # design matrix
    xi_fits = np.empty((nalphas, nbins))

    for i in range(nalphas):
        xi_fit, M = fit_to_cf_model(alpha_grid[i], xi_ls, r_avg, C_inv)
        Ms[i] = M
        xi_fits[i] = xi_fit
    
    # chi-squared test: find the alpha which minimizes chi-squared
    chi_squareds = np.ones(nalphas)
    for i in range(nalphas):
        diff = xi_ls - xi_fits[i]
        chi_squared = np.dot(diff, np.linalg.solve(C_inv, diff))
        chi_squareds[i] = chi_squared
    
    best_alpha = alpha_grid[chi_squareds.argmin()]
    M = Ms[chi_squareds.argmin()]

    return best_alpha, alpha_grid, chi_squareds, M

# loop through the list of mocks to find the best fit to cf_model for each one
def main(mock_tag = globals.mock_tag,
            data_dir = globals.data_dir,
            grad_dir = globals.grad_dir,
            grad_dim = globals.grad_dim,
            boxsize = globals.boxsize,
            density = globals.lognormal_density,
            cat_tag = globals.cat_tag,
            randmult = globals.randmult):

    s = time.time()

    mock_list_info = generate_mock_list.generate_mock_list(extra=True)  # this is only used if mock_type is not lognormal
    mock_fn_list = mock_list_info['mock_file_name_list']
    mock_param_list = mock_list_info['mock_param_list']

    if mock_tag == 'lognormal':
        cat_dir = os.path.join(data_dir, f'lognormal/xi/ls/{cat_tag}')
    else:
        cat_dir = os.path.join(grad_dir, f'ls/{cat_tag}')
    
    for i in range(len(mock_fn_list)):

        # mock_name = cat_tag if mock_tag == 'lognormal' else f'{cat_tag}_{mock_param_list[i]}'

        # load in data
        xi_results = np.load(os.path.join(cat_dir, f'xi_ls_{randmult}x_{mock_fn_list[i]}.npy'), allow_pickle=True)
        r_avg = xi_results[0]
        xi_ls = xi_results[1]

        # covariance matrix (identity for now)
        C_inv = np.identity(len(r_avg))
            # this was previously built into find_best_alpha, with option for a different C for each alpha; do I want to retain this?
            # i.e. C_invs = np.empty((nalphas, nbins, nbins))
        
        # find the best alpha (the one that minimizes chi-squared)
        best_alpha, _, _, M = find_best_alpha(xi_ls, r_avg, C_inv)
        
        # save best-fit cf
        save_dir = os.path.join(data_dir, f'bases/4-parameter_fit/results_{mock_tag}_{cat_tag}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # unpacking M matrix
        B_sq, a1, a2, a3 = M.T
        
        save_fn = os.path.join(save_dir, f'basis_{mock_tag}_{mock_fn_list[i]}')
        save_data = {
            'best_alpha' : best_alpha,
            'B_sq' : B_sq,
            'a1' : a1,
            'a2' : a2,
            'a3' : a3,
            'C_inv' : C_inv
        }
        np.save(save_fn, save_data)
        # here we only save the resulting best-fit values (as opposed to the resulting bestfit correlation function) in order to
        #   reduce redundancy and increase flexibility– B, a1, a2, and a3 can be used with any r_avg to output xi
        # should I be saving C_inv like this? we use it in patches_lstsq_fit; I want to make sure the covariances used are consistent
        
        print(f"4-parameter fit --> {mock_fn_list[i]}")


if __name__=="__main__":
    main()