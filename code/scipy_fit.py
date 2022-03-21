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

def chisq(x, A, y, Cinv):
    resid = y - A@x
    return resid @ (Cinv @ resid)

def jacobian(x, A, y, Cinv):
    resid = y - A@x
    return -2*A.T @ (Cinv @ resid)  # essentially returns the derivative of chi-squared

def scipy_fit(alpha, xi_ls, r_avg, cov, cosmo_base=None, redshift=0.57, bias=2.0):

    nbins = len(r_avg)

    # xi_mod
    xi_mod = cf_model(alpha*r_avg, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    # feature matrix
    A = np.array([xi_mod, 1/r_avg**2, 1/r_avg, np.ones(nbins)]).T

    # covariance matrix
    Cinv = np.linalg.inv(cov)      # !! this should be changed to avoid np.linalg.inv

    # initial parameters guess
    x0 = np.ones(4)

    # bounds for parameters (B^2 > 0)
    bnds = ((0, None), (None, None), (None, None), (None, None))
    
    minimize_results = scipy.optimize.minimize(chisq, x0, args=(A,xi_ls,Cinv), method='L-BFGS-B', jac=jacobian, bounds=bnds)
    M = minimize_results['x']

    # plug M (best-fit params) into xi equation
    xi_fit = A @ M

    return xi_fit, M


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

    # calculate covariance matrix
    xi_lss = np.empty((nmocks, nbins))
    for rlz in range(nmocks):
        xi_results = np.load(os.path.join(cat_dir, f'xi_ls_{randmult}x_{mock_fn_list[rlz]}.npy'), allow_pickle=True)
        r_avg = xi_results[0]
        xi_lss[rlz] = xi_results[1]
    cov = np.cov(xi_lss.T)
    
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
        
        print(f"4-parameter fit --> {mock_fn_list[i]}")
    
    total_time = time.time()-s
    print(f"total time: {datetime.timedelta(seconds=total_time)}")


if __name__=="__main__":

    for cat_tag in globals.cat_tags:
        main(cat_tag=cat_tag)