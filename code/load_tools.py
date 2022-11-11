import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

import globals
globals.initialize_vals()


def load_suave_amps(L, n, grad_dim, m, b=0.5, nmocks=401, basis='bao_fixed', data_dir=globals.data_dir, As=globals.As):
    """Return a (n,4) array of the gradient amplitudes recovered with Suave."""

    cat_tag = f'L{int(L)}_n{n}_z057_patchy_As{As}x'

    amps = np.empty((nmocks,4))
    for rlz in range(nmocks):
        suave_dict = np.load(os.path.join(data_dir, f'gradient/{grad_dim}D/suave/grad_amps/{basis}/{cat_tag}/{cat_tag}_rlz{rlz}_m-{m:.3f}-L_b-{b:.3f}.npy'), allow_pickle=True).item()
        amps[rlz] = suave_dict['amps']
    return amps


def load_patch_amps(L, n, grad_dim, m, b=0.5, nmocks=401, npatches=8, data_dir=globals.data_dir, As=globals.As):
    """Return a (n,4) array of the gradient amplitudes recovered with the standard patches approach."""

    cat_tag = f'L{int(L)}_n{n}_z057_patchy_As{As}x'

    amps = np.empty((rlzs,4))
    for rlz in range(rlzs):
        patch_dict = np.load(os.path.join(data_dir, f'gradient/{grad_dim}D/patch_data/{cat_tag}/{npatches}patches/test_dir/{cat_tag}_rlz{rlz}_m-{m:.3f}-L_b-{b:.3f}.npy'), allow_pickle=True).item()
        amps[rlz] = patch_dict['theta'].flatten()
    return amps


def grad_input(L, n, grad_dim, m, rlz=0, b=0.5, data_dir=globals.data_dir, As=globals.As):
    cat_tag = f'L{int(L)}_n{n}_z057_patchy_As{As}x'
    mock_dict = np.load(os.path.join(data_dir, f'catalogs/gradient/{grad_dim}D/{cat_tag}/{cat_tag}_rlz{rlz}_m-{m:.3f}-L_b-{b:.3f}.npy'), allow_pickle=True).item()
    return mock_dict['grad_input']


def check_grad_amps(L, n, grad_dim, m, b=0.5, nmocks=401, bins=30, alpha=0.3, data_dir=globals.data_dir, title=None, return_amps=False):
    cat_tag = f'L{L}_n{n}_z057_patchy_As2x'
    cfe_amps = load_suave_amps(L, n, grad_dim, m, b=b, nmocks=nmocks)
    
    # recovered and expected gradients
    grads_rec = np.empty((nmocks,3))
    grads_exp = np.empty(grads_rec.shape)
    for i in range(nmocks):
        grads_rec[i] = cfe_amps[i,1:]/cfe_amps[i,0]
        grads_exp[i] = grad_input(L, n, grad_dim, m, rlz=i, b=b)
    
    res = grads_rec - grads_exp
    
    # histogram of residual gradient amplitudes
    fig, ax = plt.subplots()
    
    _, bins, _ = ax.hist(res[:,0], bins=bins, alpha=alpha, label='x')
    _, _, _ = ax.hist(res[:,1], bins=bins, alpha=alpha, label='y')
    _, _, _ = ax.hist(res[:,2], bins=bins, alpha=alpha, label='y')
    
    ax.axvline(0, color='k', lw=1, alpha=0.3)
    ax.set_xlabel('Residual grad. amp. (h/Mpc)')
    ax.set_ylabel('# rlzs')
    if title:
        ax.set_title(title)
    ax.legend()

    plt.show()
    plt.close()

    if return_amps:
        data_dict = {
                    'amps' : cfe_amps,
                    'grads_rec' : grads_rec,
                    'grads_exp' : grads_exp,
                    'res' : res
                    }
        return data_dict


def compute_xi_locs(L, n, grad_dim, m, rlz, b=0.5, nvs=50, data_dir=globals.data_dir):
    from Corrfunc.utils import evaluate_xi      # (inside the function for now because of notebook issues on HPC)

    # load in suave results dictionary
    cat_tag = f'L{int(L)}_n{n}_z057_patchy_As2x'
    suave_dict = np.load(os.path.join(data_dir, f'gradient/{grad_dim}D/suave/grad_amps/bao_fixed/{cat_tag}/{cat_tag}_rlz{rlz}_m-{m:.3f}-L_b-{b:.3f}.npy'), allow_pickle=True).item()
    amps = suave_dict['amps']
    r_fine = suave_dict['r_fine']
    w_cont = suave_dict['grad_recovered']
    projfn = suave_dict['projfn']
    proj_type = suave_dict['proj_type']
    weight_type = suave_dict['weight_type']

    w_cont_hat = w_cont / np.linalg.norm(w_cont)

    # parameters
    v_min = -L/2.
    v_max = L/2.
    vs = np.linspace(v_min, v_max, nvs)
    loc_pivot = [L/2., L/2., L/2.]

    xi_locs = []
    # compute xi at nvs evenly-spaced positions across the box
    for i, v in enumerate(vs):
        loc = loc_pivot + v*w_cont_hat

        weights1 = np.array(np.concatenate(([1.0], loc-loc_pivot)))
        weights2 = weights1 # because we just take the average of these and want to get this back
    
        xi_loc = evaluate_xi(amps, r_fine, proj_type, projfn=projfn, 
                        weights1=weights1, weights2=weights2, weight_type=weight_type)    
        xi_locs.append(xi_loc)

    results_dict = {
        'vs' : vs,
        'r_fine' : r_fine,
        'xi_locs' : xi_locs
    }

    return results_dict


def save_xi_locs(L, n, grad_dim, m, rlz, b=0.5, nvs=50,
                    data_dir=globals.data_dir, save_dir='plot_data', save_fn=None):

    results = compute_xi_locs(L, n, grad_dim, m, rlz, b=b, nvs=nvs, data_dir=data_dir)

    save_fn = save_fn if save_fn else f'xi_locs_L{int(L)}_n{n}_m-{m:.3f}-L_b-{b:.3f}_{nvs}vs'

    save_path = os.path.join(data_dir, save_dir)

    np.save(os.path.join(save_path, save_fn), results, allow_pickle=True)
    print(f"calculated xi_locs and saved to {os.path.join(save_path, save_fn)}.npy")