import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import datetime

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import convert_3d_counts_to_cf
from colossus.cosmology import cosmology

from create_subdirs import create_subdirs
import generate_mock_list
from center_mock import center_mock
import globals

globals.initialize_vals()  # brings in all the default parameters

cat_tag = globals.cat_tag
grad_dim = globals.grad_dim
boxsize = globals.boxsize
lognormal_density = globals.lognormal_density
data_dir = globals.data_dir
grad_dir = globals.grad_dir
mock_type = globals.mock_type

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads


def cf_model(r, cosmo_base=None, redshift=0.0, bias=1.0):

    if cosmo_base is None:
        cosmo_base = cosmology.setCosmology('planck15')

    cf = cosmo_base.correlationFunction

    return bias**2 * cf(r, z=redshift)


# define cosmo_bases function
def cosmo_bases(rmin, rmax, projfn, cosmo_base=None, ncont=2000, 
              redshift=0.0, bias=1.0):

    if cosmo_base is None:
        print("cosmo_base not provided, defaulting to Planck 2015 cosmology ('planck15')")
        cosmo_base = cosmology.setCosmology('planck15')

    rcont = np.linspace(rmin, rmax, ncont)
    bs = cf_model(rcont, cosmo_base=cosmo_base, redshift=redshift, bias=bias)

    nbases = 1
    bases = np.empty((ncont, nbases+1))
    bases[:,0] = rcont
    bases[:,1] = np.array(bs)

    np.savetxt(projfn, bases)
    return bases


# define function to estimate gradient using suave
# cosmo=False ==> default bases are iterative results
def suave_grad(cat_tag=cat_tag, grad_dim=grad_dim, grad_dir=grad_dir, cosmo=False, plots=False):
    s = time.time()

    mock_list_info = generate_mock_list.generate_mock_list(cat_tag=cat_tag, extra=True)
    mock_file_name_list = mock_list_info['mock_file_name_list']
    mock_param_list = mock_list_info['mock_param_list']

    # make sure all inputs have the right form
    assert isinstance(grad_dim, int)
    assert isinstance(grad_dir, str)

    # create the needed subdirectories
    basis_type = 'bao_fixed' if cosmo else 'bao_iterative'
    suave_dir = f'suave_data/{cat_tag}/{basis_type}'
    plots_dir = f'plots/suave/{cat_tag}/{basis_type}/grad_recovered' if cosmo else f''

    sub_dirs = [
        suave_dir,
        plots_dir
    ]
    create_subdirs(grad_dir, sub_dirs)

    # parameters for suave (not already imported from globals)
    r_edges = np.linspace(rmin, rmax, nbins+1) 
    ncont = 2000
    r_fine = np.linspace(rmin, rmax, ncont)

    nmubins = 1
    mumax = 1.0

    proj_type = 'gradient'
    weight_type = 'pair_product_gradient'

    # basis if cosmo (outside of loop because it's the same basis for all realizations)
    if cosmo:
        projfn = os.path.join(data_dir, f'bases/bao_fixed/cosmo_basis.dat')
        basis = cosmo_bases(rmin, rmax, projfn, redshift=0.57, bias=2.0)
        ncomponents = 4*(basis.shape[1]-1)

    for i in range(len(mock_file_name_list)):

        mock_name = cat_tag if mock_type == 'lognormal' else f'{cat_tag}_{mock_param_list[i]}'

        # load bases
        if not cosmo:
            projfn = os.path.join(data_dir, f'bases/bao_iterative/results/results_gradient_{cat_tag}/final_bases/basis_gradient_{mock_name}_trrnum_{randmult}x_rlz{i}.dat')
            basis = np.loadtxt(projfn)
            ncomponents = 4*(basis.shape[1]-1)


        # load in mock and patch info
        mock_info = np.load(os.path.join(grad_dir, f'mock_data/{cat_tag}/{mock_file_name_list[i]}.npy'), allow_pickle=True).item()
        mock_file_name = mock_info['mock_file_name']
        L = mock_info['boxsize']
        mock_data = mock_info['grad_set']
        grad_expected = mock_info['grad_expected']

        # center data points between 0 and L
        center_mock(mock_data, 0, L)

        # create suave dictionary
        suave_info = {}

        nd = len(mock_data)
        x, y, z = mock_data.T

        # random set
        nr = randmult*nd
        xr = np.random.rand(nr)*float(L)
        yr = np.random.rand(nr)*float(L)
        zr = np.random.rand(nr)*float(L)

        # weights
        loc_pivot = [L/2., L/2., L/2.]
        weights = np.array([np.ones(len(x)), x-loc_pivot[0], y-loc_pivot[1], z-loc_pivot[2]])
        weights_r = np.array([np.ones(len(xr)), xr-loc_pivot[0], yr-loc_pivot[1], zr-loc_pivot[2]])

        # run the pair counts
        dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
                                proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                                periodic=periodic, weight_type=weight_type)
        # print("DD:", np.array(dd_proj))

        dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
                                X2=xr, Y2=yr, Z2=zr, weights2=weights_r, 
                                proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                                periodic=periodic, weight_type=weight_type)
        # print("DR:", np.array(dr_proj))

        rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, xr, yr, zr, weights1=weights_r, 
                                        proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                                        periodic=periodic, weight_type=weight_type)
        # print("RR:", np.array(rr_proj))

        amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
        xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

        # extract the standard binned values
        dd = np.array([x['npairs'] for x in dd_res], dtype=float)
        dr = np.array([x['npairs'] for x in dr_res], dtype=float)
        rr = np.array([x['npairs'] for x in rr_res], dtype=float)
        xi_standard = convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
        r_avg = 0.5*(r_edges[:-1] + r_edges[1:])

        # recovered gradient
        # print("amps = ", amps)
        w_cont = amps[1:]/amps[0]
        w_cont_norm = np.linalg.norm(w_cont)
        w_cont_hat = w_cont/w_cont_norm
        # print("w_cont = ", w_cont)
        # print(f"||w_cont|| = {w_cont_norm:.6f}")
        # b_guess = 0.5
        # m_recovered_perL = w_cont_norm*b_guess*L
        grad_recovered = w_cont 
        suave_info['grad_recovered'] = grad_recovered

        # print("recovered gradient (a/a_0) =", grad_recovered)

        # expected gradient (only in x direction)
        # print("expected gradient (m_input/b_input)w_hat =", grad_expected)

        # mean squared error just to see for now how close we are
        mean_sq_err = (1/len(grad_expected))*np.sum((grad_recovered - grad_expected)**2)
        # print(f"mean squared error = {mean_sq_err}")
        suave_info['mean_sq_err'] = mean_sq_err

        # non-loop-required plot parameters
        v_min = -L/2.
        v_max = L/2.
        nvs = 50
        vs = np.linspace(v_min, v_max, nvs)
        
        if plots == True:
            fig, ax = plt.subplots()
            vs_norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
            cmap = matplotlib.cm.get_cmap('cool')

        xi_locs = []

        for i, v in enumerate(vs):
            loc = loc_pivot + v*w_cont_hat
            # if i==len(vs)-1:
            #     print(loc)
            weights1 = np.array(np.concatenate(([1.0], loc-loc_pivot)))
            weights2 = weights1 #because we just take the average of these and want to get this back
            
            xi_loc = evaluate_xi(amps, r_fine, proj_type, projfn=projfn, 
                            weights1=weights1, weights2=weights2, weight_type=weight_type)    
            xi_locs.append(xi_loc)
            
            if plots == True:
                p = plt.plot(r_fine, xi_loc, color=cmap(vs_norm(v)), lw=0.5)
        
        if plots == True:
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=vs_norm)
            cbar = plt.colorbar(sm)
            cbar.set_label(r"$v \,\, (\mathbf{x} = v\hat{e}_\mathrm{gradient} + \mathbf{x}_\mathrm{pivot})$", rotation=270, labelpad=12)
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_ylim((-0.01, 0.12))
            ax.set_xlabel(r"Separation $r$ ($h^{-1}\,$Mpc)")
            ax.set_ylabel(r"$\xi(r)$")

            if mock_type == '1mock':
                ax.set_title("")
            else:
                ax.set_title(f"Recovered Gradient, {mock_file_name}")

            fig.savefig(os.path.join(grad_dir, f'{plots_dir}/{mock_file_name}.png'))

            plt.cla()
            plt.close('all')

        # save other plot parameters
        suave_info['r_avg'] = r_avg
        suave_info['amps'] = amps
        suave_info['r_fine'] = r_fine
        suave_info['xi_locs'] = xi_locs
        suave_info['proj_type'] = proj_type
        suave_info['projfn'] = projfn
        suave_info['weight_type'] = weight_type

        # save suave info dictionary
        np.save(os.path.join(grad_dir, f'{suave_dir}/{mock_file_name}'), suave_info, allow_pickle=True)

        print(f"suave --> {mock_file_name}")
    
    total_time = time.time()-s
    print(datetime.timedelta(seconds=total_time))

# function to run suave with any basis; NOT for gradient though
def suave(x, y, z, boxsize, projfn,
            proj_type='generalr',
            randmult = globals.randmult,
            nthreads=globals.nthreads,
            rmin = globals.rmin,
            rmax = globals.rmax,
            periodic = globals.periodic,
            ncont = 1000,
            nmubins = 1,
            mumax = 1.0
            ):

    # data
    data = np.array([x, y, z])
    center_mock(data, 0, boxsize)
    nd = len(x)

    # random set
    nr = randmult*nd
    xs_rand = np.random.uniform(0, boxsize, (3, nr))
    xr, yr, zr = xs_rand

    # other parameters for suave
    r_edges = np.linspace(rmin, rmax, nbins+1) 
    r_fine = np.linspace(rmin, rmax, ncont)

    # basis
    basis = np.loadtxt(projfn)
    ncomponents = basis.shape[1]-1

    # run the pair counts
    dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                                    boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                                    ncomponents=ncomponents, projfn=projfn)
    dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z,
                                    X2=xr, Y2=yr, Z2=zr,
                                    boxsize=boxsize, periodic=periodic, proj_type=proj_type,
                                    ncomponents=ncomponents, projfn=projfn)
    rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins,
                                            xr, yr, zr, boxsize=boxsize,
                                            periodic=periodic, proj_type=proj_type,
                                            ncomponents=ncomponents, projfn=projfn)
    
    amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
    xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

    # results
    results = np.empty((ncont, 2))
    results[:,0] = r_fine
    results[:,1] = xi_proj

    return results