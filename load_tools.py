import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

from generate_mock_list import MockSet
import globals
globals.initialize_vals()


def load_suave_amps(mockset, basis='bao_iterative'):
    """Return a (n,4) array of the gradient amplitudes recovered with Suave."""

    amps = np.empty((mockset.nmocks,4))
    for i, mock_fn in enumerate(mockset.mock_fn_list):
        suave_dict = np.load(os.path.join(mockset.data_dir, f'{mockset.mock_path}/suave/grad_amps/{basis}/{mock_fn}.npy'), allow_pickle=True).item()
        amps[i] = suave_dict['amps']
    return amps


def load_patch_amps(mockset, npatches=27):
    """Return a (n,4) array of the gradient amplitudes recovered with the standard patches approach."""

    amps = np.empty((mockset.nmocks,4))
    for i, mock_fn in enumerate(mockset.mock_fn_list):
        patch_dict = np.load(os.path.join(mockset.data_dir, f'{mockset.mock_path}/patches/{npatches}patches/grad_amps/{mock_fn}.npy'), allow_pickle=True).item()
        amps[i] = patch_dict['theta'].flatten()
    return amps


def grad_input(mockset, rlz=0):
    mock_dict = mockset.load_rlz(rlz)
    return mock_dict['grad_input']


def check_grad_amps(mockset, method='suave', bins=30, alpha=0.3, title=None, plot=True, return_amps=False, basis='bao_iterative', npatches=27):
    assert hasattr(mockset, 'grad_dim'), "must pass a mockset with a gradient!"

    if method=='suave':
        amps = load_suave_amps(mockset, basis=basis)
    elif method=='patches':
        amps = load_patch_amps(mockset, npatches=npatches)
    else:
        assert False, "'method' must be either 'suave' or 'patches"
    
    # recovered and expected gradients
    grads_rec = np.empty((mockset.nmocks,3))
    grads_exp = np.empty(grads_rec.shape)
    for i in range(mockset.nmocks):
        grads_rec[i] = amps[i,1:]/amps[i,0]
        grads_exp[i] = grad_input(mockset, i)
    
    res = grads_rec - grads_exp
    
    # histogram of residual gradient amplitudes
    if plot:
        fig, ax = plt.subplots()
        
        _, bins, _ = ax.hist(res[:,0], bins=bins, alpha=alpha, label='x')
        _, _, _ = ax.hist(res[:,1], bins=bins, alpha=alpha, label='y')
        _, _, _ = ax.hist(res[:,2], bins=bins, alpha=alpha, label='z')
        
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
                    'amps' : amps,
                    'grads_rec' : grads_rec,
                    'grads_exp' : grads_exp,
                    'res' : res
                    }
        return data_dict


def t_intersect(omega, n, a, O):
    # ensure unit vectors
    n /= np.linalg.norm(n)
    omegahat = omega / np.linalg.norm(omega)
    aprime = np.array(a) - np.array(O)
    A = np.dot(n, aprime)
    B = np.dot(n, omegahat)
    if B==0:
        print("this line and plane are parallel!")
        t = np.inf
    else:
        t = np.dot(n, aprime) / np.dot(n, omegahat)
    return t


def compute_v(omega, L, O=[0,0,0]):
    """Computes the maximum distance along the gradient direction."""
    omegahat = omega / np.linalg.norm(omega)  # ensure unit vector
    amag = L/2  # distance to each box face
    # test intersection with yz-plane first
    comps = {0 : 'yz',
             1 : 'xz',
             2 : 'xy'}
    ts = []
    for i in comps:
        # construct unit vector
        n = np.zeros(3)
        n[i] = 1
        a = amag * n
        t = t_intersect(omegahat, n, a, O)  # "time" to intersect with the plane (=box face)
        ts.append(np.abs(t))
        print(f"t = {t:.1f} for the {comps[i]}-plane")
    v = min(ts)  # the way we parameterize means that t is just the physical distance to the box edge
    comps_intersect = np.where(ts==v)[0][0]  # will be just one component unless omega is parallel to some face
    print(f"line will intersect first with the {comps[comps_intersect]} plane: v={v:.2f}")
    return v


def compute_xi_locs(mockset, i, nvs=50, basis='bao_iterative'):
    from Corrfunc.utils import evaluate_xi

    # unpack mockset params
    data_dir = mockset.data_dir
    mock_path = mockset.mock_path
    mock_fn = mockset.mock_fn_list[i]
    L = mockset.L

    # load in suave results dictionary
    suave_dict = np.load(os.path.join(data_dir, f'{mock_path}/suave/grad_amps/{basis}/{mock_fn}.npy'), allow_pickle=True).item()

    amps = suave_dict['amps']
    r_fine = suave_dict['r_fine']
    projfn = suave_dict['projfn']
    proj_type = suave_dict['proj_type']
    weight_type = suave_dict['weight_type']

    w_cont = amps[1:]/amps[0]
    print(f"w_cont = {w_cont}")
    w_cont_hat = w_cont / np.linalg.norm(w_cont)

    # parameters
    v_max = compute_v(w_cont, L)
    print(f"v_max = {v_max:.1f}")
    v_min = -v_max
    vs = np.linspace(v_min, v_max, nvs)
    loc_pivot = [L/2., L/2., L/2.]  # center of the box

    xi_locs = []
    # compute xi at nvs evenly-spaced positions across the box
    for i, v in enumerate(vs):
        # print(loc_pivot, v, w_cont_hat)
        loc = loc_pivot + v*w_cont_hat

        weights1 = np.array(np.concatenate(([1.0], loc-loc_pivot)))
        weights2 = weights1 # because we just take the average of these and want to get this back
    
        xi_loc = evaluate_xi(amps, r_fine, proj_type, projfn=projfn, 
                        weights1=weights1, weights2=weights2, weight_type=weight_type)    
        xi_locs.append(xi_loc)

    results_dict = {
        'w_cont' : w_cont,
        'vs' : vs,
        'r_fine' : r_fine,
        'xi_locs' : xi_locs,
    }

    return results_dict


def save_xi_locs(mockset, rlz, nvs=50, basis='bao_iterative', save_dir='plot_data', save_fn=None):

    results = compute_xi_locs(mockset, rlz, nvs=nvs, basis=basis)

    save_fn = save_fn if save_fn else f'xi_locs_{mockset.mock_fn_list[rlz]}_{nvs}vs'

    save_path = os.path.join(mockset.data_dir, save_dir)

    np.save(os.path.join(save_path, save_fn), results, allow_pickle=True)
    print(f"calculated xi_locs and saved to {os.path.join(save_path, save_fn)}.npy")