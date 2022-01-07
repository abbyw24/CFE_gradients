import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

from suave import cf_model

import generate_mock_list

import globals

def fit_to_cf_model(alpha, xi_ls, r_avg, ncont=1000):

    nbins = len(r_avg)
    cosmo_base = None
    redshift = 0.57
    bias = 2.0

    # xi_mod
    xi_mod = cf_model(alpha*r_avg, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    # matrices
    Y = xi_ls       # results of L-S estimator on entire mock (i.e. no patches); vector of length nbins

    A = np.array([xi_mod, 1/r_avg**2, 1/r_avg, np.ones(nbins)]).T       # feature matrix

    C = np.identity(len(Y))     # covariance matrix; ones for now
    C_inv = np.linalg.inv(C)

    # performing the fit
    X = np.linalg.inv(A.T @ C_inv @ A) @ (A.T @ C_inv @ Y)

    B_sq, a1, a2, a3 = X

    xi_fit = B_sq*xi_mod + a1/r_avg**2 + a2/r_avg + a3

    # # putting X values back into equation for xi_fit— but with finer r grid
    # r_avg_fine = np.linspace(min(r_avg), max(r_avg), ncont+1)
    # xi_mod_fine = cf_model(alpha*r_avg_fine, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    # xi_fit = B_sq*xi_mod_fine + a1/r_avg_fine**2 + a2/r_avg_fine + a3

    return xi_fit, X, C

# for the standard approach, we perform this fit for several different alpha values and find the one which minimizes chi-squared
def find_best_alpha(xi_ls, r_avg, alpha_min=0.75, alpha_max=1.25, nalphas=51):

    nbins = len(r_avg)
    alpha_grid = np.linspace(alpha_min, alpha_max, nalphas)

    xi_fits = np.empty((nalphas, nbins))
    Cs = np.empty((nalphas, nbins, nbins))
    for i in range(nalphas):
        xi_fit, _, C = fit_to_cf_model(alpha_grid[i], xi_ls, r_avg)
        xi_fits[i] = xi_fit
        Cs[i] = C
    
    # chi-squared test: find the alpha which minimizes chi-squared
    chi_squareds = np.ones(nalphas)
    for i in range(nalphas):
        diff = xi_ls - xi_fits[i]
        chi_squared = np.dot(diff, np.linalg.solve(Cs[i], diff))
        chi_squareds[i] = chi_squared
    
    best_alpha = alpha_grid[chi_squareds.argmin()]

    return best_alpha, alpha_grid, chi_squareds

# loop through the list of mocks to find the best fit to cf_model for each one
def main():
    s = time.time()

    globals.initialize_vals()
    mock_tag = globals.mock_tag
    data_dir = globals.data_dir
    grad_dir = globals.grad_dir
    grad_dim = globals.grad_dim
    boxsize = globals.boxsize
    density = globals.lognormal_density
    cat_tag = globals.cat_tag
    randmult = globals.randmult

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
        
        # find the best alpha (the one that minimizes chi-squared)
        best_alpha, _, _ = find_best_alpha(xi_ls, r_avg)

        # plug best_alpha back in to retrieve the best-fit correlation function
        xi_bestfit, _, _ = fit_to_cf_model(best_alpha, xi_ls, r_avg)
        
        # save best-fit cf
        save_dir = os.path.join(data_dir, f'bases/4-parameter_fit/results_{mock_tag}_{cat_tag}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = os.path.join(save_dir, f'basis_{mock_tag}_{mock_fn_list[i]}')
        np.save(save_fn, [r_avg, xi_bestfit])
            # should I save other values such as alpha or X (bestfit) values?
        
        print(f"4-parameter fit --> {mock_fn_list[i]}")


if __name__=="__main__":
    main()