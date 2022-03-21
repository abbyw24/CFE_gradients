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

nmocks = globals.nmocks
nbins = globals.nbins

def scipy_fit(alpha, xi_ls, r_avg, cov, cosmo_base=None, redshift=0.57, bias=2.0):

    nbins = len(r_avg)

    # xi_mod
    xi_mod = cf_model(alpha*r_avg, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    # define function to be optimized by scipy
    def func(r, B, a1, a2, a3):
        xi_bestfit = B**2*xi_mod + a1/r**2 + a2/r + a3
        return xi_bestfit
    
    params, _ = scipy.optimize.curve_fit(func, r_avg, xi_ls, sigma=cov)
    B, a1, a2, a3 = params
    M = np.array([B**2, a1, a2, a3])
    scipy_fit = func(r_avg, B, a1, a2, a3)

    return scipy_fit, M


def find_best_alpha(xi_ls, r_avg, cov, alpha_min=0.75, alpha_max=1.25, nalphas=501):

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


# loop through the list of mocks to find the best fit to cf_model for each one
def main(mock_tag = globals.mock_tag,
            data_dir = globals.data_dir,
            grad_dir = globals.grad_dir,
            grad_dim = globals.grad_dim,
            boxsize = globals.boxsize,
            density = globals.lognormal_density,
            cat_tag = globals.cat_tag,
            randmult = globals.randmult,
            cov_type = 'full'):

    s = time.time()

    mock_list_info = generate_mock_list.generate_mock_list(extra=True)  # this is only used if mock_type is not lognormal
    mock_fn_list = mock_list_info['mock_file_name_list']
    mock_param_list = mock_list_info['mock_param_list']

    if mock_tag == 'lognormal':
        cat_dir = os.path.join(data_dir, f'lognormal/xi/ls/{cat_tag}')
    else:
        cat_dir = os.path.join(grad_dir, f'ls/{cat_tag}')

    # calculate covariance matrix
    if cov_type == 'identity':
        cov = np.identity(nbins)
    elif cov_type =='full':
        xi_lss = np.empty((nmocks, nbins))

        for rlz in range(nmocks):
            xi_results = np.load(os.path.join(cat_dir, f'xi_ls_{randmult}x_{mock_fn_list[rlz]}.npy'), allow_pickle=True)
            r_avg = xi_results[0]
            xi_lss[rlz] = xi_results[1]

            ls_std = np.std(xi_lss, axis=0)

        cov = np.cov(xi_lss.T)
    else:
        assert False, "unknown cov_type"
    
    for i in range(len(mock_fn_list)):

        # load in data
        xi_results = np.load(os.path.join(cat_dir, f'xi_ls_{randmult}x_{mock_fn_list[i]}.npy'), allow_pickle=True)
        r_avg = xi_results[0]
        xi_ls = xi_results[1]
        
        # find the best alpha (the one that minimizes chi-squared)
        best_alpha, _, min_chi_squared, M = find_best_alpha(xi_ls, r_avg, cov)
        
        # save best-fit cf
        save_dir = os.path.join(data_dir, f'bases/4-parameter_fit/scipy/results_{mock_tag}_{cat_tag}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # unpacking M matrix
        B_sq, a1, a2, a3 = M.T
        
        save_fn = os.path.join(save_dir, f'basis_{mock_tag}_{mock_fn_list[i]}')
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
        
        print(f"4-parameter fit --> {mock_fn_list[i]}: {cov_type} covariance matrix")
    
    total_time = time.time()-s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


if __name__=="__main__":
    main(cov_type='full')