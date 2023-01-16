import numpy as np
import os
import sys
from scipy import special, integrate

import generate_mock_list
import globals
globals.initialize_vals()


# Roman's equation for the VARIANCE of xi at a given r
def xi_err_pred(r, V, n, kG, PkG):
    const = (2*np.pi)**5 / V
    # function of k that we want to integrate
    def k_func(k, Pk):
        return (k/r) * (Pk+(1/n))**2 * (special.jv(1/2, k*r))**2
    # construct our array to integrate
    k_func_arr = np.array([k_func(k, PkG[i]) for i, k in enumerate(kG)])
    trapz = integrate.trapezoid(k_func_arr, x=kG)
    return const*trapz


def compute_predicted_xi_err(L=globals.boxsize, n=globals.lognormal_density, As=globals.As):

    # initialize mock set
    mock_set = generate_mock_list.MockSet(L, n, As=As)

    # input power spectrum
    PkG_fn = f'/scratch/ksf293/mocks/lognormal/inputs/cat_{mock_set.cat_tag}_pkG.dat'
    kG, PkG = np.loadtxt(PkG_fn).T

    # load in r values to use
    mock_set.load_xi_lss()

    # get predicted variances using xi_err_pred()
    predicted_vars = [xi_err_pred(r, L**3, float(n), kG, PkG) for r in mock_set.r_avg]

    # save predicted variances
    save_arr = np.array([mock_set.r_avg, predicted_vars]).T
    save_fn = os.path.join(globals.data_dir, f'predicted_xi_var_{mock_set.cat_tag}.npy')
    np.save(save_fn, save_arr)
    print(f"saved predicted xi variance ---> {save_fn}")


if __name__=='__main__':

    compute_predicted_xi_err()