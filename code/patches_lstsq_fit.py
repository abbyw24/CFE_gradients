import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import os
import time
import datetime

import generate_mock_list
from center_mock import center_mock
from suave import cf_model
from corrfunc_ls import compute_ls
import globals

globals.initialize_vals()  # brings in all the default parameters


"""
INPUTS:
- mock galaxy catalog
- 4-parameter fit from scipy_fit.py (the realization-specific bases analogous to bao_iterative)
- binned L-S results in each patch

RETURNS:
- best-fit parameters (a0, ax, ay, az) of the gradient
"""


def xi_bestfit(r, xi_mod, B_sq, a1, a2, a3):

    xi_bestfit = B_sq*xi_mod + a1/r**2 + a2/r + a3

    return xi_bestfit


# load in results from the 4-parameter fit (analogous to bao_iterative)
def f_bases(basis_fn, r, x):

    # load in best-fit parameters
    fit_results = np.load(basis_fn, allow_pickle=True).item()
    alpha = fit_results['best_alpha']
    B_sq = fit_results['B_sq']
    a1 = fit_results['a1']
    a2 = fit_results['a2']
    a3 = fit_results['a3']
    cov = fit_results['cov']

    # xi_model: the fiducial cf model at the best-fit alpha
    xi_model = cf_model(alpha*r, cosmo_base=None, redshift=0.57, bias=2.0)

    # plug best_alpha back in to retrieve the best-fit correlation function at this separation
    xi_fit = xi_bestfit(r, xi_model, B_sq, a1, a2, a3)

    x_pos = np.array([1, x[0], x[1], x[2]])
    
    return xi_fit * x_pos, cov


def patchify(data, boxsize, npatches=globals.npatches):
    """Divide a mock catalog into n cubic patches."""
    n_sides = npatches**(1/3)
    assert n_sides.is_integer()
    n_sides = int(n_sides)
    nd, n_dim = data.shape
    boxsize_patch = boxsize/n_sides
    a = np.arange(n_sides)
    idx_patches = np.array(list(it.product(a, repeat=n_dim)))
    patch_ids = np.zeros(nd).astype(int)
    for patch_id,iii in enumerate(idx_patches):
        # define the boundaries of the patch
        mins = iii*boxsize_patch
        maxes = (iii+1)*boxsize_patch
        # define mask as where all of the values must be within the boundaries
        mask_min = np.array([(data[:,d]<maxes[d]) for d in range(n_dim)])
        mask_max = np.array([(data[:,d]>=mins[d]) for d in range(n_dim)])
        mask = np.vstack([mask_min, mask_max]).T
        mask_combined = np.all(mask, axis=1)
        # perform masking and save to array
        patch_ids[mask_combined] = patch_id
    return patch_ids, idx_patches


def xi_in_patches(x, y, z, L, n,
            npatches = globals.npatches,
            load_rand = True,
            data_dir = globals.data_dir,
            randmult = globals.randmult,
            periodic = globals.periodic,
            rmin = globals.rmin,
            rmax = globals.rmax,
            nbins = globals.nbins,
            nthreads = globals.nthreads
            ):
    """Compute the Landy-Szalay 2pcf in each of n patches of a single mock catalog."""
    
    mock_data = np.array([x,y,z]).T
    nd = len(x)

    # random set: either load a pre-computed set, or generate one here
    if load_rand:
        try:
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{int(L)}_n{n}_{randmult}x.dat')
        except OSError: # generate the random catalog if it doesn't already exist
            random_cat.main(L, n, data_dir, randmult)
            random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{int(L)}_n{n}_{randmult}x.dat')
        finally:
            rand_set = np.loadtxt(random_fn)
        # rand_set.shape == (nr, 3)
    else:
        nr = randmult * nd
        rand_set = np.random.uniform(0, L, (int(nr),3))
    center_mock(rand_set, 0, L)

    # patchify mock data and random set
    patches_mock = patchify(mock_data, L, npatches=npatches)
    patch_ids_mock = patches_mock[0]
    patch_id_list_mock = np.unique(patch_ids_mock)

    patches_rand = patchify(rand_set, L, npatches=npatches)
    patch_ids_rand = patches_rand[0]
    patch_id_list_rand = np.unique(patch_ids_rand)

    # make sure patch lists match for mock and random
    assert np.all(patch_id_list_mock == patch_id_list_rand)
    patch_id_list = patch_id_list_mock
    npatches = len(patch_id_list)
    patches_idx = patches_mock[1]

    # define patch centers by taking mean of the random set in each patch
    patch_centers = np.empty((npatches, 3))
    for i, patch_id in enumerate(patch_id_list):
        patch_data = rand_set[patch_ids_rand == patch_id]
        center = np.mean(patch_data, axis=0)
        patch_centers[i] = center

    # results in patches
    xi_patches = np.empty((npatches, nbins))

    for i, patch_id in enumerate(patch_id_list):
        patch_data = mock_data[patch_ids_mock == patch_id]
        patch_rand = rand_set[patch_ids_rand == patch_id]

        r_avg, xi_thispatch = compute_ls(patch_data, patch_rand, periodic, nthreads, rmin, rmax, nbins)

        xi_patches[i] = xi_thispatch

    xi_patches = np.array(xi_patches)

    # save xi results to the dictionary
    patch_dict = {
        'patch_centers' : patch_centers,
        'r_avg' : r_avg,
        'xi_patches' : xi_patches
    }

    return patch_dict


# def patches_lstsq_fit(cat_tag=globals.cat_tag,
#                         grad_dim=globals.grad_dim,
#                         grad_dir=globals.grad_dir,
#                         npatches=globals.n_patches,
#                         mock_range=None, test=False, cov_type='diag'):
    
#     s = time.time()

#     mock_file_name_list = generate_mock_list.generate_mock_list(cat_tag=cat_tag)

#     mock_range = mock_range if mock_range else range(len(mock_file_name_list))

#     for i in mock_range:
#         mock_info = np.load(os.path.join(grad_dir, f'mock_data/{cat_tag}/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
#         mock_fn = mock_info['mock_file_name']
#         L = mock_info['boxsize']

#         patch_info = np.load(os.path.join(grad_dir, f'patch_data/{cat_tag}/{npatches}patches/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
#         patch_centers = patch_info['patch_centers']
#         # patch_centers.shape == (npatches, 3)

#         # center mock around 0
#         center_mock(patch_centers, -L/2, L/2)

#         r = patch_info['r_avg']
#         nbins = len(r)
#         xi_patches = patch_info['xi_patches']
#         # xi_patches.shape == (npatches, nbins)

#         # below we "unroll" the array with binned xi in each patch (dimensions nbins x npatches) into a 1D array with length nbins x npatches
#         #   in order to perform our least square fit

#         # create empty arrays
#         X = np.empty((nbins*npatches, 4))
#         xi = np.empty((nbins*npatches, 1))

#         # fill the arrays with the relevant data
#         for patch in range(npatches):
#             for nbin in range(nbins):
#                 X_bin, cov = f_bases(mock_fn, r[nbin], patch_centers[patch])

#                 xi_bin = xi_patches[patch, nbin]
#                 row = patch*nbins + nbin
#                 X[row,:] = X_bin
#                 xi[row] = xi_bin
    
#         # "tile" covariance array (since we now have nbins x npatches)
#         cov_tiled = np.tile(cov, (npatches,npatches))

#         if cov_type == 'diag':
#             cov_big = np.zeros(cov_tiled.shape)
#             np.fill_diagonal(cov_big, np.diag(cov_tiled))
#         elif cov_type == 'full':
#             cov_big = cov_tiled
#         else:
#             assert False, "cov_type must be 'diag' or 'full'"

#         # performing the fit: xi = X @ theta

#         a = X.T @ np.linalg.solve(cov_big, X)
#         b = X.T @ np.linalg.solve(cov_big, xi)

#         # a theta = b
#         theta, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    
#         # add recovered values to patch info dictionary
#         patch_info['theta'] = theta

#         # add recovered gradient value to patch info dictionary
#         grad_recovered = theta[1:]/theta[0]
#         patch_info['grad_recovered'] = grad_recovered

#         # change back patch_center values for dictionary saving
#         center_mock(patch_centers, 0, L)
#         # resave patch info dictionary
#         np.save(os.path.join(grad_dir, f'patch_data/{cat_tag}/{npatches}patches/{mock_fn}'), patch_info, allow_pickle=True)

#         print(f"lstsqfit in {npatches} patches --> {mock_fn}")
    
#     total_time = time.time()-s
#     print(f"fit to patches --> {cat_tag}, {len(mock_range)} mocks")
#     print(f"total time: {datetime.timedelta(seconds=total_time)}")


def compute_patches_lstsqfit(x, y, z, L, n, basis_fn,
                        npatches = globals.npatches,
                        load_rand = True,
                        data_dir = globals.data_dir,
                        randmult = globals.randmult,
                        periodic = globals.periodic,
                        rmin = globals.rmin,
                        rmax = globals.rmax,
                        nbins = globals.nbins,
                        nthreads = globals.nthreads):
    """Compute a simultaneous least-squares fit to each bin in each patch for a single mock catalog."""

    # create patches dictionary
    patch_dict = {}

    # compute Landy-Szalay in each patch
    xi_patch_results = xi_in_patches(x, y, z, L, n,
                                    npatches=npatches, load_rand=load_rand, data_dir=data_dir, randmult=randmult, periodic=periodic,
                                    rmin=rmin, rmax=rmax, nbins=nbins, nthreads=nthreads)

    # unpack dictionary results
    patch_centers = xi_patch_results['patch_centers']
    r_avg = xi_patch_results['r_avg']
    xi_patches = xi_patch_results['xi_patches']

    # below we "unroll" the array with binned xi in each patch (dimensions nbins x npatches) into a 1D array with length nbins x npatches
    #   in order to perform our least square fit

    # create empty arrays
    X = np.empty((nbins*npatches, 4))
    xi = np.empty((nbins*npatches, 1))

    # fill the arrays with the relevant data
    for patch in range(npatches):
        for nbin in range(nbins):
            X_bin, cov = f_bases(basis_fn, r_avg[nbin], patch_centers[patch])

            xi_bin = xi_patches[patch, nbin]
            row = patch*nbins + nbin
            X[row,:] = X_bin
            xi[row] = xi_bin
    
    # "tile" covariance array (since we now have nbins x npatches)
    cov_tiled = np.tile(cov, (npatches,npatches))
    # and take only the diagonals (the full covariance matrix throws a singular matrix error)
    cov_big = np.zeros(cov_tiled.shape)
    np.fill_diagonal(cov_big, np.diag(cov_tiled))

    # performing the fit: xi = X @ theta

    a = X.T @ np.linalg.solve(cov_big, X)
    b = X.T @ np.linalg.solve(cov_big, xi)

    # a @ theta = b
    theta, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    patch_dict['theta'] = theta
    patch_dict['patch_centers'] = patch_centers
    patch_dict['xi_patches'] = xi_patches
    patch_dict['r_avg'] = r_avg
    patch_dict['cov'] = cov_big
    patch_dict['load_rand'] = load_rand
    patch_dict['randmult'] = randmult
    patch_dict['periodic'] = periodic
    patch_dict['basis_fn'] = basis_fn

    return patch_dict