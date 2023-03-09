import numpy as np
import multiprocessing as mp
from itertools import repeat
from time import sleep

# import gradmock_gen
# import xi_funcs
# import fetch_lognormal_mocks
import load_tools


# # parameters for mp
# # Ls = [500, 750, 1000, 1500]
# L = 750
# n = '1e-4'
# grad_dim = 1
# # input_w = [np.sqrt(3), 1, 0]
# ms = [0.1, 0.25, 0.5, 0.75, 1]

# defining new function to fit starmap() inputs
def run_in_parallel(L, n, grad_dim, m, rlz):
    load_tools.compute_xi_locs(L, n, grad_dim, m, rlz)
    # gradmock_gen.generate_gradmocks(L=L, n=n, grad_dim=grad_dim, m=m)
    # xi_funcs.xi_ls_mocklist(L=L, n=n, grad_dim=grad_dim, m=m)
    # xi_funcs.grad_cfe_mocklist(L=L, n=n, grad_dim=grad_dim, m=m)


def mp_starmap(func_to_run, args):
    with mp.Pool() as pool:
        mp_res = pool.starmap(func_to_run, args)
        mp_res
    
    # shut down mp properly
    sleep(1)
    pool.close()
    pool.join()
    sleep(1)


def fetch_mocks(rlzs):
    fetch_lognormal_mocks.fetch_ln_mocks('L750_n2e-4_z057_patchy', rlzs)


def mp_processes(func_to_run, rlzs, nprocesses=10):
    sets = np.array_split(rlzs, nprocesses)
    # sets = np.array(sets).T     # transpose so that when these sets run in parallel, the smaller idns still run before the larger ones
    print(f"chunks:")
    for chunk in sets:
        print(chunk)

    procs = []
    for i, rlz_set in enumerate(sets):
        print(f"starting set {i+1} of {len(sets)}...")
        proc = mp.Process(target=func_to_run, args=(rlz_set,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


def main():
    # parameters
    L = 750
    n = '2e-4'
    grad_dim = 1
    ms = [0.1, 0.25, 0.5, 0.75, 1]
    rlz = 0

    mp_starmap(run_in_parallel, zip(repeat(L), repeat(n), repeat(grad_dim), ms, repeat(rlz)))
    # mp_processes(fetch_mocks, rlzs=np.arange(500, 1000), nprocesses=10)


if __name__=='__main__':
    main()