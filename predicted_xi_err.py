import numpy as np
import os
import sys
from scipy import special, integrate

import generate_mock_list
import globals
globals.initialize_vals()


# Roman's equation for the VARIANCE of xi at a given r
def xi_err_pred(r, V, n, k, Pk, return_extra=False):

    # multiplicative constant
    const = 1 / (2*np.pi*V) # (2*np.pi)**5 / V

    # function of k that we want to integrate
    def k_func(k, Pk):
        # return (k/r) * (Pk+(1/n))**2 * (special.jv(1/2, k*r))**2
        return (k/r) * (Pk+(1/((2*np.pi)**3 *n)))**2 * (special.jv(1/2, k*r))**2

    # construct our array, and integrate using trapezoid rule
    k_func_arr = np.array([k_func(k, Pk[i]) for i, k in enumerate(k)])
    trapz = integrate.trapezoid(k_func_arr, x=k)

    # optionally, return extra intermediate results
    if return_extra:
        result_dict = {
            'predicted_var' : const*trapz,
            'const' : const,
            'integral' : trapz,
            'k' : k,
            'k_func_arr' : k_func_arr
        }
        return result_dict
    else:
        return const*trapz


def compute_predicted_xi_err_fixed_n(L=globals.boxsize, n=globals.lognormal_density, As=globals.As, kmin=None, kmax=None, bias=2.0):

    # initialize mock set
    mock_set = generate_mock_list.MockSet(L, n, As=As)

    # input power spectrum
    # PkG_fn = f'/scratch/ksf293/mocks/lognormal/inputs/cat_{mock_set.cat_tag}_pkG.dat'
    # k, Pk = np.loadtxt(PkG_fn).T
    Pk_fn = f'/scratch/ksf293/mocks/lognormal/inputs/cat_{mock_set.cat_tag}_pk.txt'
    k, Pk = np.loadtxt(Pk_fn).T

    # convert from matter power spectrum to galaxy power spectrum
    PkG = bias**2 * Pk
    
    print("input power spectrum:")
    print(len(k), len(Pk))

    # optionally, only integrate over part of the input power spectrum in k_func()
    idx = (k>kmin)&(k<kmax)
    k = k[idx]
    PkG = PkG[idx]
    print("after cuts:")
    print(len(k), len(PkG))

    # load in r values to use
    mock_set.load_xi_lss()

    # get predicted variances using xi_err_pred(), along with extra info
    predicted_vars = np.empty(mock_set.nbins)
    consts = np.empty(mock_set.nbins)
    integrals = np.empty(mock_set.nbins)
    k_func_arrs = np.empty((mock_set.nbins, len(k)))
    for i, r in enumerate(mock_set.r_avg):
        result_thisr = xi_err_pred(r, L**3, float(n), k, PkG, return_extra=True)
        predicted_vars[i] = result_thisr['predicted_var']
        consts[i] = result_thisr['const']
        integrals[i] = result_thisr['integral']
        k_func_arrs[i] = result_thisr['k_func_arr']

    # save predicted variances
    save_dict = {
        'n' : n,
        'r_avg' : mock_set.r_avg,
        'predicted_vars' : predicted_vars,
        'k' : k,
        'PkG' : PkG,
        'consts' : consts,
        'integrals' : integrals,
        'k_func_arrs' : k_func_arrs
    }
    ktag = f'k-{kmin:.0f}-{kmax:.0f}'
    save_fn = os.path.join(globals.data_dir, f'pred_xi_var_n{n}_L{L}_{ktag}_v3.npy')
    np.save(save_fn, save_dict)
    print(f"saved predicted xi variance ---> {save_fn}")


def compute_predicted_xi_err_fixed_r(ns, r_idx=10, L=globals.boxsize, As=globals.As, kmin=None, kmax=None, bias=2.0):

    # initialize mock set, with the first number density (we only do this to get the r values)
    mock_set = generate_mock_list.MockSet(L, '2e-4', As=As)
    # load in r values to use
    mock_set.load_xi_lss()
    r = mock_set.r_avg[r_idx]
    print(f"computing variance in xi at r = {r:.2f} Mpc/h:")

    # input power spectrum (does not depend on number density)
    Pk_fn = f'/scratch/ksf293/mocks/lognormal/inputs/cat_{mock_set.cat_tag}_pk.txt'
    k, Pk = np.loadtxt(Pk_fn).T
    # convert from matter power spectrum to galaxy power spectrum
    PkG = bias**2 * Pk
    print("input power spectrum:")
    print(len(k), len(Pk))
    # optionally, only integrate over part of the input power spectrum in k_func()
    idx = (k>kmin)&(k<kmax)
    k = k[idx]
    PkG = PkG[idx]
    print("after cuts:")
    print(len(k), len(PkG))

    # get predicted variances using xi_err_pred(), along with extra info
    predicted_vars = np.empty(len(ns))
    consts = np.empty(len(ns))
    integrals = np.empty(len(ns))
    k_func_arrs = np.empty((len(ns), len(k)))
    for i, n in enumerate(ns):
        result_thisr = xi_err_pred(r, L**3, float(n), k, PkG, return_extra=True)
        predicted_vars[i] = result_thisr['predicted_var']
        consts[i] = result_thisr['const']
        integrals[i] = result_thisr['integral']
        k_func_arrs[i] = result_thisr['k_func_arr']

    # save predicted variances
    save_dict = {
        'r' : r,
        'ns' : ns,
        'predicted_vars' : predicted_vars,
        'k' : k,
        'PkG' : PkG,
        'consts' : consts,
        'integrals' : integrals,
        'k_func_arrs' : k_func_arrs
    }
    ktag = f'k-{kmin:.0f}-{kmax:.0f}'
    save_fn = os.path.join(globals.data_dir, f'pred_xi_var_r{r:.2f}_L{L}_{ktag}_v3.npy')
    np.save(save_fn, save_dict)
    print(f"saved predicted xi variance ---> {save_fn}")


if __name__=='__main__':

    ns = np.logspace(-6,-3,50)
    compute_predicted_xi_err_fixed_r(ns, r_idx=2, kmin=.008, kmax=20)

    # compute_predicted_xi_err_fixed_n(kmin=.008, kmax=20)