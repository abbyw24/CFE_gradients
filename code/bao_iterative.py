import os
import numpy as np
import matplotlib.pyplot as plt
import glob 
import re
import time
import datetime

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import trr_analytic
from Corrfunc.bases import bao_bases
from colossus.cosmology import cosmology

import read_lognormal as reader

from center_mock import center_mock
import generate_mock_list
import globals
globals.initialize_vals()

def main(mock_type=globals.mock_type,
            L=globals.boxsize, n=globals.lognormal_density, As=globals.As,
            data_dir=globals.data_dir, rlzs=globals.rlzs,
            grad_dim=globals.grad_dim, m=globals.m, b=globals.b, same_dir=globals.same_dir,
            rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins,
            randmult=globals.randmult, nthreads=globals.nthreads, load_rand=True, prints=False):

    s = time.time()
    
    # generate the mock set parameters
    mock_set = generate_mock_list.MockSet(L, n, As=As, data_dir=data_dir, rlzs=rlzs)
    cat_tag = mock_set.cat_tag
    rlzs = mock_set.rlzs

    # check whether we want to use gradient mocks or lognormal mocks
    if mock_type=='gradient':
        mock_set.add_gradient(grad_dim, m, b, same_dir=same_dir)
    else:
        assert mock_type=='lognormal', "mock_type must be either 'gradient' or 'lognormal'"

    if load_rand:
        random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{L}_n{n}_{randmult}x.dat')
    else:
        random_fn = None

    proj = 'baoiter'
    # cosmo_name options: ['b17', 'planck', 'wmap9'] (for example)
    # cosmo_name = 'b17'
    cosmo_name = 'planck15'
    cosmo = cosmology.setCosmology(cosmo_name)
    cf_tag = f"_{proj}_cosmo{cosmo_name}_test"
    redshift = 0.57

    restart_unconverged = False # will restart any unconverged realizations - WARNING, IF FALSE WILL OVERWRITE ITERATIONS OF UNCONVERGED ONES
    convergence_threshold = 1e-5 #when to stop (fractional change)
    niter_max = 160 # stop after this many iterations if not converged 

    save_iterated_bases = True
    skip_converged = True
    trr_analytic = False
    periodic = False
    dalpha = 0.001
    k0 = 0.1

    # cosmo = bao_utils.get_cosmo(cosmo_name)
    trr_tag = '' if trr_analytic else '_trrnum'
    rand_tag = '' if trr_analytic else f'_{randmult}x'
    per_tag = '_perTrue' if periodic else ''

    for Nr, rlz in enumerate(rlzs):
        if prints:
            print(f"Realization {rlz}")

        alpha_model_start = 1.0
        eta = 0.5
        biter = BAO_iterator(Nr, rlz, mock_type, mock_set, rand_tag, cosmo, L, periodic,
                            grad_dim=grad_dim, rmin=rmin, rmax=rmax, nbins=nbins, cf_tag=cf_tag, trr_analytic=trr_analytic,
                            save_iterated_bases=save_iterated_bases, nthreads=nthreads, redshift=redshift,
                            alpha_model_start=alpha_model_start, dalpha=dalpha, k0=k0, random_fn=random_fn)

        # update this
        if skip_converged:
            converged_fn = f'{biter.result_dir}/converged/cf{biter.cf_tag}_{biter.trr_tag}{biter.rand_tag}{biter.per_tag}_{biter.mock_fn}.npy'
            matches = glob.glob(converged_fn)
            if len(matches)>0:
                print("Already converged and saved to", matches[0])
                continue

        # initial parameters
        niter_start = 0
        if restart_unconverged:
            pattern = f"cf{biter.cf_tag}_{trr_tag}{rand_tag}{per_tag}_niter([0-9]+)_{biter.mock_fn}.npy"
            niters_done = []
            for fn in os.listdir(biter.result_dir):
                matches = re.search(pattern, fn)
                if matches is not None:
                    niters_done.append(int(matches.group(1)))
            
            if niters_done: # if matches found (list not empty), start from latest iter; otherwise, will start from zero
                niter_lastdone = max(niters_done) # load in last completed one
                start_fn = f'{biter.result_dir}/cf{biter.cf_tag}_{trr_tag}{rand_tag}{per_tag}_niter{niter_lastdone}_{biter.mock_fn}.npy'
                res = np.load(start_fn, allow_pickle=True, encoding='latin1')
                _, _, amps, _, extra_dict = res
                alpha_model_prev = extra_dict['alpha_model']
                C = amps[4]
                alpha_model_start = alpha_model_prev + eta*C*k0
                niter_start = niter_lastdone + 1
        
        if prints:
            print(f"Starting from iteration {niter_start}")
        # set up iterative procedure
        biter.load_catalogs()
        alpha_model = alpha_model_start

        niter = niter_start
        err = np.inf
        err_prev = np.inf
        alpha_result_prev = np.inf
        converged = False
        while (not converged) and (niter < niter_max):

            xi, amps, bases = biter.bao_iterative(dalpha, alpha_model, niter)
            C = amps[4]

            alpha_result = alpha_model + C*k0
            err = (alpha_result - alpha_result_prev)/alpha_result
            extra_dict = {'r_edges': biter.rbins, 'ncomponents': biter.ncomponents, 
                          'proj_type': biter.proj_type, 'projfn': biter.projfn,
                          'alpha_start': alpha_model_start, 'alpha_model': alpha_model,
                          #'alpha_model_next': alpha_model_next, #for if iteration interrupted
                          'dalpha': dalpha, 'alpha_result': alpha_result,
                          'niter': niter, 'err': err}

            if prints:
                print(f'iter {niter}')
            # print(f'alpha: {alpha_model}, dalpha: {dalpha}')
            # print(f"C: {C}")
            if abs(err) < convergence_threshold:
                converged = True
            
            biter.save_cf(xi, amps, niter, extra_dict, rand_tag=rand_tag, converged=converged, prints=prints)

            # update alphas
            c1 = err>0
            c2 = err_prev>0
            if np.isfinite(err) and np.isfinite(err_prev) and (c1 != c2):
                # if the error switched sign, reduce eta
                eta *= 0.75
            # print("Adaptive eta:", err, err_prev, eta)
            # print(alpha_model,alpha_model + eta*C*k0)           
            alpha_model = alpha_model + eta*C*k0
           
            alpha_result_prev = alpha_result
            err_prev = err

            niter += 1
            
            if prints:
                print(f'NEW alpha: {alpha_model}, dalpha: {dalpha}')
                print(f'err: {err} (threshold: {convergence_threshold})')

            if alpha_model < 0:
                assert False, "new alpha is negative; check input parameters!"

        # Declare why stopped
        if niter==niter_max:
            print(f"hit max number of iterations, {niter_max}")
        if converged:
            if prints:
                print(f"converged after {niter} iterations with error {err} (threshold {convergence_threshold})")

            # resave converged correlation function (our new basis function!) as a .dat file (instead of .npy) to fit suave script
            biter.save_final_basis(xi, prints=prints)
    
    total_time = time.time()-s
    print(f"bao_iterative --> {biter.result_dir}/final_bases, {mock_set.nmocks} mocks")
    print(f"total time: {datetime.timedelta(seconds=total_time)}")
            


class BAO_iterator:

    def __init__(self, Nr, rlz, mock_type, mock_set, rand_tag, cosmo, boxsize, periodic, 
                    grad_dim=globals.grad_dim, rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, cf_tag='_baoiter',
                    trr_analytic=False, save_iterated_bases=False, nthreads=globals.nthreads, redshift=0.0, bias=2.0,
                    alpha_model_start=1.0, dalpha=0.01, k0=0.1, random_fn=None):

        # input params
        self.Nr = Nr
        self.rlz = rlz
        self.mock_type = mock_type
        self.rand_tag = rand_tag
        self.cosmo = cosmo
        self.boxsize = boxsize
        self.periodic = periodic

        self.grad_dim = grad_dim
        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.cf_tag = cf_tag
        
        self.trr_analytic = trr_analytic
        self.save_iterated_bases = save_iterated_bases
        self.nthreads = nthreads
        self.redshift = redshift
        self.bias = bias
        self.alpha_model_start = alpha_model_start
        self.dalpha = dalpha
        self.k0 = k0
        self.random_fn = random_fn

        # other params
        self.mumax = 1.0
        self.weight_type = None
        self.nmubins = 1
        self.verbose = False
        self.proj_type = 'generalr'
        # set up r params
        self.rbins = np.linspace(rmin, rmax, nbins+1)
        self.rbins_avg = 0.5*(self.rbins[1:]+self.rbins[:-1])
        self.rcont = np.linspace(rmin, rmax, 2000)
        # and random set params
        self.trr_tag = 'trrana' if self.trr_analytic else 'trrnum'
        self.per_tag = '_perTrue' if self.periodic else ''
        if not trr_analytic and random_fn is None:
            raise ValueError("Must choose trr_analytic or pass random_fn!")

        # mock_set parameters
        self.cat_tag = mock_set.cat_tag
        self.data_dir = mock_set.data_dir
        self.mock_fn = mock_set.mock_fn_list[Nr]
        self.cat_dir = os.path.join(self.data_dir, f'catalogs/{mock_set.mock_path}')

        self.bases_dir = os.path.join(self.data_dir, f'bases/bao_iterative/{mock_set.mock_path}')

        self.projfn = os.path.join(self.bases_dir, f"tables/bases_{self.mock_type}_{self.mock_fn}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}.dat")
        if not os.path.exists(os.path.join(self.bases_dir, f"tables")):
            os.makedirs(os.path.join(self.bases_dir, f"tables"))

        # set up result dir
        self.result_dir = os.path.join(self.bases_dir, f'results')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # write initial bases
        projfn_start = os.path.join(self.bases_dir, f"tables/bases_{self.mock_type}_{self.mock_fn}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}.dat")
        #alpha_guess was previously called alpha_model
        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_guess':alpha_model_start, 'bias':self.bias}
        #self.ncomponents, _ = bao.write_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)
        # print(os.getcwd())
        bases = bao_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)
        base_vals = bases[:,1:]
        self.ncomponents = base_vals.shape[1]


    def save_cf(self, xi, amps, niter, extra_dict, rand_tag, converged=True, prints=False):
        if converged:
            save_dir = f'{self.result_dir}/converged'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_fn = os.path.join(save_dir, f'cf{self.cf_tag}_{self.trr_tag}{self.rand_tag}{self.per_tag}_{self.mock_fn}.npy')
            np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])
            if prints:
                print(f"Saved converged to {save_fn}")

            # if mocks are lognormal, also save converged cf to the lognormal result directory
            if self.mock_type == 'lognormal':
                save_dir = f'{self.data_dir}/lognormal/{self.cat_tag}/xi/bao_iterative'
                save_fn = os.path.join(save_dir, f'xi{self.cf_tag}_{self.trr_tag}{self.rand_tag}{self.per_tag}_{self.mock_fn}.npy')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])
                if prints:
                    print(f"Saved converged to {save_fn}")
        else:
            save_fn = f'{self.result_dir}/cf{self.cf_tag}_{self.trr_tag}{self.rand_tag}{self.per_tag}_niter{niter}_{self.mock_fn}.npy'
            np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])

        
    def save_final_basis(self, xi, prints=False):
        if not os.path.exists(os.path.join(self.result_dir, 'final_bases')):
            os.makedirs(os.path.join(self.result_dir, f'final_bases'))
        basis_fn = f'{self.result_dir}/final_bases/basis_{self.mock_fn}_{self.trr_tag}{self.rand_tag}.dat'
        final_basis = np.array([self.rcont, xi]).T
        np.savetxt(basis_fn, final_basis)
        if prints:
            print(f"Saved final basis to {basis_fn}")


    def load_catalogs(self):
        self.load_data()
        if not self.trr_analytic:
            self.load_random()


    def load_data(self):
        data_fn = os.path.join(self.cat_dir, f'{self.mock_fn}.npy')
        mock_info = np.load(data_fn, allow_pickle=True).item()
        L, N, data = mock_info['L'], mock_info['N'], mock_info['data']
        assert float(L) == float(self.boxsize)
        center_mock(data, 0, self.boxsize)
        self.x, self.y, self.z = data.T
        self.nd = N
        self.weights = None


    def load_random(self):
        random = np.loadtxt(self.random_fn)
        center_mock(random, 0, self.boxsize)
        self.x_rand, self.y_rand, self.z_rand = random.T
        self.nr = random.shape[0]
        self.weights_rand = None


    def run_estimator_analytic(self):
        # TODO: make that can pass rbins as None to DDsmu for e.g. generalr when dont need!

        _, dd_proj, _ = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                        proj_type=self.proj_type, ncomponents=self.ncomponents, projfn=self.projfn,
                        verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic, isa='fallback')
        projfn = np.loadtxt(self.projfn)

        volume = float(self.boxsize**3)
        rr_ana, trr_ana = trr_analytic(self.rmin, self.rmax, self.nd, volume, self.ncomponents, self.proj_type, rbins=self.rbins, projfn=self.projfn)
    
        numerator = dd_proj - rr_ana
        amps_periodic_ana = np.linalg.solve(trr_ana, numerator)
        # print("AMPS:", amps_periodic_ana)
        xi_periodic_ana = evaluate_xi(amps_periodic_ana, self.rcont, self.proj_type, rbins=self.rbins, projfn=self.projfn)

        return xi_periodic_ana, amps_periodic_ana


    def run_estimator_numeric(self):

        # print("running DD")
        # print(f"nthreads: {self.nthreads}, rbins: {self.rbins}, mumax: {self.mumax}, nmubins: {self.nmubins}")
        # print(f"x: {self.x}, y: {self.y}, z: {self.z}")
        # print(f"proj_type: {self.proj_type}, ncomponents: {self.ncomponents}, projfn: {self.projfn}, verbose: {self.verbose}, boxsize: {self.boxsize}, periodic: {self.periodic}")
        _, dd_proj, _ = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                        proj_type=self.proj_type, ncomponents=self.ncomponents, projfn=self.projfn,
                        verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)

        _, dr_proj, _ = DDsmu(0, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                            X2=self.x_rand, Y2=self.y_rand, Z2=self.z_rand, 
                            proj_type=self.proj_type, ncomponents=self.ncomponents, projfn=self.projfn,
                            verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)

        _, rr_proj, trr_proj = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x_rand, self.y_rand, self.z_rand,
               proj_type=self.proj_type, ncomponents=self.ncomponents, projfn=self.projfn,
               verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)
        
        # print("nd nr", self.nd, self.nr)
        amps = compute_amps(self.ncomponents, self.nd, self.nd, self.nr, self.nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
        # print("AMPS:", amps)
        xi_proj = evaluate_xi(amps, self.rcont, self.proj_type, rbins=self.rbins, projfn=self.projfn)

        return xi_proj, amps


    # is this a good or an ugly way to do this toggle?
    def run_estimator(self):
        if self.trr_analytic:
            return self.run_estimator_analytic()
        else:
            return self.run_estimator_numeric()


    def bao_iterative(self, dalpha, alpha_model, niter):

        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_guess':alpha_model, 'bias':self.bias, 'k0':self.k0}
        # self.ncomponents, _ = bao.write_bases(self.rbins[0], self.rbins[-1], self.projfn, **kwargs)    
        bases = bao_bases(self.rbins[0], self.rbins[-1], self.projfn, **kwargs)

        # save bases
        if self.save_iterated_bases:
            save_dir = os.path.join(self.bases_dir, 'tables/iterated_bases')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_basis_fn = os.path.join(save_dir, f'bases_{self.mock_type}_{self.mock_fn}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}_niter{niter}.dat')
            np.savetxt(save_basis_fn, bases)

        base_vals = bases[:,1:]
        xi, amps = self.run_estimator()

        return xi, amps, bases



if __name__=="__main__":
    
    main()