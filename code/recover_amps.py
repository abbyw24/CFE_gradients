import numpy as np
import os
import globals
globals.initialize_vals()

def recover_amps(mock_file_name_list, cat_tag, grad_dir=globals.grad_dir, nbins=globals.nbins, n_patches=8, method="suave", basis=None):
    amps_rec = np.empty((len(mock_file_name_list), 4))
    xi_full_patches = np.zeros((len(mock_file_name_list), nbins))
    if method == "suave":
        for i in range(len(mock_file_name_list)):
            suave_info = np.load(os.path.join(grad_dir, f"suave_data/{cat_tag}/{basis}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            amps = suave_info["amps"]
            amps_rec[i] = amps
        return amps_rec
    elif method == "patches":
        for i in range(len(mock_file_name_list)):
            patch_info = np.load(os.path.join(grad_dir, f"patch_data/{cat_tag}/{n_patches}patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            b_fit, m_fit = patch_info["b_fit"], patch_info["m_fit"].flatten()
            amps_rec[i,0], amps_rec[i,1:] = b_fit, m_fit
            xi = patch_info["xi_full"]
            xi_full_patches[i] = xi
        return amps_rec, xi_full_patches
    else:
        print("")