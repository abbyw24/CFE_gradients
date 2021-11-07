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

def main():
    s = time.time()

    globals.initialize_vals()
    mock_type = globals.mock_type
    data_dir = globals.data_dir
    grad_dim = globals.grad_dim
    boxsize = globals.boxsize
    density = globals.lognormal_density
    cat_tag = globals.cat_tag
    n_mocks = globals.n_mocks
    randmult = globals.randmult
    
    data_dir = '/scratch/aew492/research-summer2020_output'
    mock_list_info = generate_mock_list.generate_mock_list(extra=True)  # this is only used if mock_type is not lognormal

    if mock_type == 'lognormal':
        cat_dir = os.path.join(data_dir, f'catalogs/lognormal/cat_{cat_tag}') #f'/scratch/ksf293/mocks/lognormal/cat_{cat_tag}'
    else:
        cat_dir = os.path.join(data_dir, f'gradient/{grad_dim}D/mock_data/{cat_tag}')

    random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{boxsize}_n{density}_{randmult}x.dat')  # generate my own random catalogs

    proj = 'baoiter'
    # cosmo_name options: ['b17', 'planck', 'wmap9'] (for example)
    # cosmo_name = 'b17'
    cosmo_name = 'planck15'
    cosmo = cosmology.setCosmology(cosmo_name)
    cf_tag = f"_{proj}_cosmo{cosmo_name}_test"
    redshift = 0.57
    realizations = range(n_mocks)

    restart_unconverged = False # will restart any unconverged realizations - WARNING, IF FALSE WILL OVERWRITE ITERATIONS OF UNCONVERGED ONES
    convergence_threshold = 1e-5 #when to stop (fractional change)
    niter_max = 160 # stop after this many iterations if not converged 

    save_iterated_bases = True
    skip_converged = False
    trr_analytic = False     # needs to be False eventually
    periodic = False
    nthreads = globals.nthreads
    dalpha = 0.001
    k0 = 0.1

    print(Corrfunc.__file__)
    print(Corrfunc.__version__)

    #cosmo = bao_utils.get_cosmo(cosmo_name)
    trr_tag = '' if trr_analytic else '_trrnum'
    rand_tag = '' if trr_analytic else f'_{randmult}x'
    per_tag = '_perTrue' if periodic else ''

    for Nr in realizations:
        print(f"Realization {Nr}")

        alpha_model_start = 1.0
        eta = 0.5
        biter = BAO_iterator(mock_type, boxsize, periodic, cat_tag, rand_tag, cat_dir, cosmo, data_dir,
                            mock_list_info=mock_list_info, Nr=Nr, cf_tag=cf_tag, trr_analytic=trr_analytic,
                            save_iterated_bases=save_iterated_bases, nthreads=nthreads, redshift=redshift,
                            alpha_model_start=alpha_model_start, dalpha=dalpha, k0=k0, random_fn=random_fn)

        # update this
        if skip_converged:
            converged_fn = f'{biter.result_dir}/converged/cf{biter.cf_tag}_{biter.trr_tag}{biter.rand_tag}{biter.per_tag}_{biter.grad_tag}_rlz{biter.Nr}.npy'
            matches = glob.glob(converged_fn)
            if len(matches)>0:
                print("Already converged and saved to", matches[0])
                continue

        # initial parameters
        niter_start = 0
        if restart_unconverged:
            pattern = f"cf{biter.cf_tag}{trr_tag}{rand_tag}{per_tag}_niter([0-9]+)_{biter.grad_tag}_rlz{biter.Nr}.npy"
            niters_done = []
            for fn in os.listdir(biter.result_dir):
                matches = re.search(pattern, fn)
                if matches is not None:
                    niters_done.append(int(matches.group(1)))
            
            if niters_done: # if matches found (list not empty), start from latest iter; otherwise, will start from zero
                niter_lastdone = max(niters_done) # load in last completed one
                start_fn = f'{biter.result_dir}/cf{biter.cf_tag}{trr_tag}{rand_tag}{per_tag}_niter{niter_lastdone}_{biter.grad_tag}_rlz{biter.Nr}.npy'
                res = np.load(start_fn, allow_pickle=True, encoding='latin1')
                _, _, amps, _, extra_dict = res
                alpha_model_prev = extra_dict['alpha_model']
                C = amps[4]
                alpha_model_start = alpha_model_prev + eta*C*k0
                niter_start = niter_lastdone + 1
        
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
            extra_dict = {'r_edges': biter.rbins, 'ncomponents': biter.ncomponents, 
                          'proj_type': biter.proj_type, 'projfn': biter.projfn,
                          'alpha_start': alpha_model_start, 'alpha_model': alpha_model,
                          #'alpha_model_next': alpha_model_next, #for if iteration interrupted
                          'dalpha': dalpha, 'alpha_result': alpha_result,
                          'niter': niter}

            print(f'iter {niter}')
            # print(f'alpha: {alpha_model}, dalpha: {dalpha}')
            # print(f"C: {C}")
            err = (alpha_result - alpha_result_prev)/alpha_result
            if abs(err) < convergence_threshold:
                converged = True
            
            biter.save_cf(xi, amps, niter, extra_dict, rand_tag=rand_tag, converged=converged)

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
            
            print(f'NEW alpha: {alpha_model}, dalpha: {dalpha}')
            print(f'err: {err} (threshold: {convergence_threshold})')

            if alpha_model < 0:
                assert False, "new alpha is negative; check input parameters!"

        # Declare why stopped
        if niter==niter_max:
            print(f"hit max number of iterations, {niter_max}")
        if converged:
            print(f"converged after {niter} iterations with error {err} (threshold {convergence_threshold})")

            # resave converged correlation function (our new basis function!) as a .dat file (instead of .npy) to fit suave script
            biter.save_final_basis(xi)
    
    total_time = time.time()-s
    print(datetime.timedelta(seconds=total_time))
            


class BAO_iterator:

    def __init__(self, mock_type, boxsize, periodic, cat_tag, rand_tag, cat_dir, cosmo, data_dir, mock_list_info=None, Nr=0,
                    rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, cf_tag='_baoiter', trr_analytic=False,
                    save_iterated_bases=False, nthreads=globals.nthreads, redshift=0.0, bias=2.0,
                    alpha_model_start=1.0, dalpha=0.01, k0=0.1, random_fn=None):

        # input params
        self.mock_type = mock_type
        self.boxsize = boxsize
        self.Nr = Nr
        self.cosmo = cosmo

        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.redshift = redshift

        # other params
        self.mumax = 1.0
        self.bias = bias
        self.k0 = k0
        self.weight_type = None
        self.periodic = periodic
        self.nthreads = nthreads
        self.nmubins = 1
        self.verbose = False
        self.proj_type = 'generalr'
        self.trr_analytic = trr_analytic
        self.save_iterated_bases = save_iterated_bases

        # set up other data
        self.rbins = np.linspace(rmin, rmax, nbins+1)
        self.rbins_avg = 0.5*(self.rbins[1:]+self.rbins[:-1])
        self.rcont = np.linspace(rmin, rmax, 2000)

        if mock_type != 'lognormal':
            self.mock_list_info = mock_list_info
            self.mock_fn_list = self.mock_list_info['mock_file_name_list']
            self.grad_tag_list = self.mock_list_info['grad_tag_list']

        self.cat_tag = cat_tag
        self.grad_tag = cat_tag if self.mock_type == 'lognormal' else f'{cat_tag}_{self.grad_tag_list[self.Nr]}'
        self.rand_tag = rand_tag
        self.cat_dir = cat_dir
        self.cf_tag = cf_tag
        self.trr_tag = 'trrana' if self.trr_analytic else 'trrnum'
        self.per_tag = '_perTrue' if self.periodic else ''
        if not trr_analytic and random_fn is None:
            raise ValueError("Must choose trr_analytic or pass random_fn!")
        self.random_fn = random_fn

        self.data_dir = data_dir
        self.bases_dir = os.path.join(self.data_dir, 'bases/bao_iterative')
        self.mock_tag = 'lognormal' if mock_type == 'lognormal' else 'gradient'
        self.projfn = os.path.join(self.bases_dir, f"tables/bases_{self.mock_tag}_{self.grad_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}_rlz{self.Nr}.dat")
        if not os.path.exists(os.path.join(self.bases_dir, f"tables")):
            os.makedirs(os.path.join(self.bases_dir, f"tables"))

        # set up result dir
        self.result_dir = os.path.join(self.bases_dir, 'results/results_{}_{}'.format(self.mock_tag, self.cat_tag))
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # write initial bases
        projfn_start = os.path.join(self.bases_dir, f"tables/bases_{self.grad_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}.dat")
        #alpha_guess was previously called alpha_model
        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_guess':alpha_model_start, 'bias':self.bias}
        #self.ncomponents, _ = bao.write_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)
        # print(os.getcwd())
        bases = bao_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)
        base_vals = bases[:,1:]
        self.ncomponents = base_vals.shape[1]


    # currently only for lognormal mock_type
    def save_cf(self, xi, amps, niter, extra_dict, rand_tag, converged=True):
        if converged:
            if self.mock_tag == 'lognormal':
                save_dir = f'{self.data_dir}/lognormal/xi/bao_iterative/{self.cat_tag}'
                save_fn = os.path.join(save_dir, f'xi{self.cf_tag}_{self.trr_tag}{self.rand_tag}{self.per_tag}_{self.cat_tag}_rlz{self.Nr}.npy')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])
                print(f"Saved converged to {save_fn}")

            save_dir = f'{self.result_dir}/converged'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_fn = os.path.join(save_dir, f'cf{self.cf_tag}_{self.trr_tag}{self.rand_tag}{self.per_tag}_{self.grad_tag}_rlz{self.Nr}.npy')
            np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])
            print(f"Saved converged to {save_fn}")
        else:
            save_fn = f'{self.result_dir}/cf{self.cf_tag}_{self.trr_tag}{self.rand_tag}{self.per_tag}_niter{niter}_{self.grad_tag}_rlz{self.Nr}.npy'
            np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])

        
    def save_final_basis(self, xi):
        if not os.path.exists(os.path.join(self.result_dir, 'final_bases')):
            os.makedirs(os.path.join(self.result_dir, f'final_bases'))
        basis_fn = f'{self.result_dir}/final_bases/basis_{self.mock_tag}_{self.grad_tag}_{self.trr_tag}{self.rand_tag}_rlz{self.Nr}.dat'
        final_basis = np.array([self.rcont, xi]).T
        np.savetxt(basis_fn, final_basis)
        print(f"Saved final basis to {basis_fn}")


    def load_catalogs(self):
        self.load_data()
        if not self.trr_analytic:
            self.load_random()


    def load_data(self):
        if self.mock_tag == 'lognormal':
            data_fn = f'{self.cat_dir}/cat_{self.cat_tag}_lognormal_rlz{self.Nr}.bin'
            L, _, _, N, data = reader.read(data_fn) # first 3 are Lx, Ly, Lz
            self.x, self.y, self.z, _, _, _ = data.T
            assert L == self.boxsize, "boxsize from data must equal boxsize from globals"
            pos = np.array([self.x, self.y, self.z])
            center_mock(pos, 0, self.boxsize)
            self.nd = N
        else:   # different data structure to load for gradient mocks
            assert self.mock_tag == 'gradient'
            data_fn = f'{self.cat_dir}/{self.mock_fn_list[self.Nr]}.npy'
            mock_info = np.load(data_fn, allow_pickle=True).item()
            L, N, data = mock_info['boxsize'], mock_info['N'], mock_info['grad_set']
            center_mock(data, 0, self.boxsize)
            self.x, self.y, self.z = data.T
            # data_fn = f'{self.data_dir}/catalogs/gradient/{self.cat_tag}/{self.mock_fn_list[self.Nr]}.npy'
            # data = np.load(data_fn, allow_pickle=True)
            # center_mock(data, 0, self.boxsize)
            # self.x, self.y, self.z = data.T
            self.nd = len(data)

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
            save_basis_fn = os.path.join(save_dir, f'bases_{self.mock_tag}_{self.grad_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}_rlz{self.Nr}_niter{niter}.dat')
            np.savetxt(save_basis_fn, bases)

        base_vals = bases[:,1:]
        xi, amps = self.run_estimator()

        return xi, amps, bases



if __name__=="__main__":
    main()