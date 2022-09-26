import numpy as np
import os

import globals
globals.initialize_vals()


def load_suave_amps(cat_tag, grad_dim, m, b=0.5, rlzs=401, basis='bao_fixed', data_dir=globals.data_dir):
    """Return a (n,4) array of the gradient amplitudes recovered with Suave."""
    amps = np.empty((rlzs,4))
    for rlz in range(rlzs):
        suave_dict = np.load(os.path.join(data_dir, f'gradient/{grad_dim}D/suave_data/{cat_tag}/{basis}/{cat_tag}_rlz{rlz}_m-{m:.3f}-L_b-{b:.3f}.npy'), allow_pickle=True).item()
        amps[rlz] = suave_dict['amps']
    return amps


def load_patch_amps(cat_tag, m, b=0.5, rlzs=401, npatches=8, grad_dir=globals.grad_dir):
    """Return a (n,4) array of the gradient amplitudes recovered with the standard patches approach."""
    amps = np.empty((rlzs,4))
    for rlz in range(rlzs):
        patch_dict = np.load(os.path.join(grad_dir, f'patch_data/{cat_tag}/{npatches}patches/test_dir/{cat_tag}_rlz{rlz}_m-{m:.3f}-L_b-{b:.3f}.npy'), allow_pickle=True).item()
        amps[rlz] = patch_dict['theta'].flatten()
    return amps