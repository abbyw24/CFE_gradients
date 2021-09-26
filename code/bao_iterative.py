import os
import numpy as np
import matplotlib.pyplot as plt
import glob 
import re

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import trr_analytic
from Corrfunc.bases import bao_bases
from colossus.cosmology import cosmology

import read_lognormal as reader

from center_mock import center_mock

import globals
globals.initialize_vals()


def main():

    boxsize = globals.boxsize
    density = globals.lognormal_density
    cat_tag = f'_L{boxsize}_n{density}_z057_patchy_As2x'
    data_dir = '/scratch/aew492/research-summer2020_output'
    result_dir = os.path.join(data_dir, f'lognormal/iterative_results/lognormal{cat_tag}')
    cat_dir = f'/scratch/ksf293/mocks/lognormal/cat{cat_tag}'
    random_fn = os.path.join(data_dir, f'catalogs/randoms/rand_L{boxsize}_n{density}_1x.dat')  # generate my own random catalogs

    proj = 'baoiter'
    # cosmo_name options: ['b17', 'planck', 'wmap9'] (for example)
    # cosmo_name = 'b17'
    cosmo_name = 'planck15'
    cosmo = cosmology.setCosmology(cosmo_name)
    cf_tag = f"_{proj}_cosmo{cosmo_name}_test"
    redshift = 0.57
    realizations = [0]

    restart_unconverged = True # will restart any unconverged realizations - WARNING, IF FALSE WILL OVERWRITE ITERATIONS OF UNCONVERGED ONES
    convergence_threshold = 1e-3 #when to stop (fractional change)
    niter_max = 160 # stop after this many iterations if not converged 

    skip_converged = True
    trr_analytic = False     # needs to be False eventually
    nthreads = globals.nthreads
    dalpha = 0.1    # previously .001
    k0 = 0.1

    print(Corrfunc.__file__)
    print(Corrfunc.__version__)

    #cosmo = bao_utils.get_cosmo(cosmo_name)
    trr_tag = '' if trr_analytic else '_trrnum'

    for Nr in realizations:
        print(f"Realization {Nr}")

        if skip_converged:
            converged_fn = f'{result_dir}/cf{cf_tag}{trr_tag}_converged_niter*{cat_tag}_rlz{Nr}.npy'
            matches = glob.glob(converged_fn)
            if len(matches)>0:
                print("Already converged and saved to", matches[0])
                continue

        alpha_model_start = 1.0
        eta = 0.5
        biter = BAO_iterator(boxsize, cat_tag, cat_dir, cosmo, data_dir, Nr=Nr, cf_tag=cf_tag, trr_analytic=trr_analytic, nthreads=nthreads, redshift=redshift, alpha_model_start=alpha_model_start, dalpha=dalpha, k0=k0, random_fn=random_fn)

        # initial parameters
        niter_start = 0
        if restart_unconverged:
            pattern = f"cf{biter.cf_tag}{trr_tag}_niter([0-9]+){biter.cat_tag}_rlz{biter.Nr}.npy"
            niters_done = []
            for fn in os.listdir(biter.result_dir):
                matches = re.search(pattern, fn)
                if matches is not None:
                    niters_done.append(int(matches.group(1)))
            
            if niters_done: # if matches found (list not empty), start from latest iter; otherwise, will start from zero
                niter_lastdone = max(niters_done) # load in last completed one
                start_fn = f'{biter.result_dir}/cf{biter.cf_tag}{trr_tag}_niter{niter_lastdone}{biter.cat_tag}_rlz{biter.Nr}.npy'
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

            xi, amps, bases = biter.bao_iterative(dalpha, alpha_model)
            C = amps[4]

            alpha_result = alpha_model + C*k0
            extra_dict = {'r_edges': biter.rbins, 'ncomponents': biter.ncomponents, 
                          'proj_type': biter.proj_type, 'projfn': biter.projfn,
                          'alpha_start': alpha_model_start, 'alpha_model': alpha_model,
                          #'alpha_model_next': alpha_model_next, #for if iteration interrupted
                          'dalpha': dalpha, 'alpha_result': alpha_result,
                          'niter': niter}

            print(f'iter {niter}')
            print(f'alpha: {alpha_model}, dalpha: {dalpha}')
            print(f"C: {C}")
            err = (alpha_result - alpha_result_prev)/alpha_result
            if abs(err) < convergence_threshold:
                converged = True
            biter.save_cf(xi, amps, niter, extra_dict, converged=converged)

            # update alphas
            c1 = err>0
            c2 = err_prev>0
            if np.isfinite(err) and np.isfinite(err_prev) and (c1 != c2):
                # if the error switched sign, reduce eta
                eta *= 0.75
            print("Adaptive eta:", err, err_prev, eta)
            print(alpha_model,alpha_model + eta*C*k0)           
            alpha_model = alpha_model + eta*C*k0
           
            alpha_result_prev = alpha_result
            err_prev = err

            niter += 1
            
            print(f'NEW alpha: {alpha_model}, dalpha: {dalpha}')
            print(f'err: {err} (threshold: {convergence_threshold})')

        # Declare why stopped
        if niter==niter_max:
            print(f"hit max number of iterations, {niter_max}")
        if converged:
            print(f"converged after {niter} iterations with error {err} (threshold {convergence_threshold})")

            # resave converged correlation function (our new basis function!) as a .dat file (instead of .npy) to fit suave script
            biter.save_basis(xi, amps, niter, extra_dict)
            


class BAO_iterator:

    def __init__(self, boxsize, cat_tag, cat_dir, cosmo, data_dir, Nr=0, rmin=globals.rmin, rmax=globals.rmax, nbins=globals.nbins, 
                 cf_tag='_baoiter', trr_analytic=False, nthreads=globals.nthreads, 
                 redshift=0.0, bias=2.0, alpha_model_start=1.0, dalpha=0.01, k0=0.1,
                 random_fn=None):

        # input params
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
        self.periodic = False   # previously True
        self.nthreads = nthreads
        self.nmubins = 1
        self.verbose = False
        self.proj_type = 'generalr'
        self.trr_analytic = trr_analytic

        # set up other data
        self.rbins = np.linspace(rmin, rmax, nbins+1)
        self.rbins_avg = 0.5*(self.rbins[1:]+self.rbins[:-1])
        self.rcont = np.linspace(rmin, rmax, 2000)

        self.cat_tag = cat_tag
        self.cat_dir = cat_dir
        self.cf_tag = cf_tag
        if not trr_analytic and random_fn is None:
            raise ValueError("Must choose trr_analytic or pass random_fn!")
        self.random_fn = random_fn

        self.data_dir = data_dir
        self.projfn = os.path.join(self.data_dir, f"tables/bases{self.cat_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}_rlz{self.Nr}.dat")
        if not os.path.exists(os.path.join(self.data_dir, f"tables")):
            os.makedirs(os.path.join(self.data_dir, f"tables"))

        # set up result dir
        self.result_dir = os.path.join(self.data_dir, 'results/results_lognormal{}'.format(self.cat_tag))
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # write initial bases
        projfn_start = os.path.join(self.data_dir, f"tables/bases{self.cat_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}.dat")
        #alpha_guess was previously called alpha_model
        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_guess':alpha_model_start, 'bias':self.bias}
        #self.ncomponents, _ = bao.write_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)
        print(os.getcwd())
        bases = bao_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)
        base_vals = bases[:,1:]
        # print("base vals:", base_vals)
        self.ncomponents = base_vals.shape[1]


    def save_cf(self, xi, amps, niter, extra_dict, converged=True):
        trr_tag = '' if self.trr_analytic else '_trrnum'
        conv_tag = '_converged' if converged else ''
        save_fn = f'{self.result_dir}/cf{self.cf_tag}{trr_tag}{conv_tag}_niter{niter}{self.cat_tag}_rlz{self.Nr}.npy'
        np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])
        print(f"Saved to {save_fn}")

    def save_basis(self, xi, amps, niter, extra_dict):
        if not os.path.exists(os.path.join(self.data_dir, f'tables/final_bases')):
            os.makedirs(os.path.join(self.data_dir, f'tables/final_bases'))
        basis_fn = f'{self.data_dir}/tables/final_bases/basis{self.cat_tag}_rlz{self.Nr}.dat'
        basis = np.array([self.rcont, xi]).T
        np.savetxt(basis_fn, basis)
        print(f"Saved final basis to {basis_fn}")


    def load_catalogs(self):
        self.load_data()
        if not self.trr_analytic:
            self.load_random()


    def load_data(self):
        data_fn = f'{self.cat_dir}/cat{self.cat_tag}_lognormal_rlz{self.Nr}.bin'
        L, _, _, N, data = reader.read(data_fn) # first 3 are Lx, Ly, Lz
        assert L == self.boxsize, "boxsize from data must equal boxsize from globals"
        self.x, self.y, self.z, _, _, _ = data.T
        pos = np.array([self.x, self.y, self.z])
        center_mock(pos, 0, self.boxsize)
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
        print("projfn:", self.projfn, projfn)
        print("dd:", dd_proj)

        volume = float(self.boxsize**3)
        rr_ana, trr_ana = trr_analytic(self.rmin, self.rmax, self.nd, volume, self.ncomponents, self.proj_type, rbins=self.rbins, projfn=self.projfn)
    
        numerator = dd_proj - rr_ana
        amps_periodic_ana = np.linalg.solve(trr_ana, numerator)
        print("AMPS:", amps_periodic_ana)
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
        
        print("nd nr", self.nd, self.nr)
        amps = compute_amps(self.ncomponents, self.nd, self.nd, self.nr, self.nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
        print("AMPS:", amps)
        xi_proj = evaluate_xi(amps, self.rcont, self.proj_type, rbins=self.rbins, projfn=self.projfn)

        return xi_proj, amps


    # is this a good or an ugly way to do this toggle?
    def run_estimator(self):
        if self.trr_analytic:
            return self.run_estimator_analytic()
        else:
            return self.run_estimator_numeric()


    def bao_iterative(self, dalpha, alpha_model):

        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_guess':alpha_model, 'bias':self.bias, 'k0':self.k0}
        # self.ncomponents, _ = bao.write_bases(self.rbins[0], self.rbins[-1], self.projfn, **kwargs)    
        bases = bao_bases(self.rbins[0], self.rbins[-1], self.projfn, **kwargs)
        xi, amps = self.run_estimator()

        return xi, amps, bases



if __name__=="__main__":
    main()