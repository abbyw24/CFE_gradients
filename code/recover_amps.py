import numpy as np
import os
import globals

def recover_amps(mock_file_name_list, num_den, nbins=globals.nbins, n_patches=8, method="suave"):
    amps_rec = np.empty((len(mock_file_name_list), 4))
    xi_full_patches = np.zeros((len(mock_file_name_list), nbins))
    abs_path = "/scratch/aew492/research-summer2020_output/1D/"
    if method == "suave":
        for i in range(len(mock_file_name_list)):
            suave_info = np.load(os.path.join(abs_path, f"suave_data/{num_den}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            amps = suave_info["amps"]
            amps_rec[i] = amps
        return amps_rec
    elif method == "patches":
        for i in range(len(mock_file_name_list)):
            patch_info = np.load(os.path.join(abs_path, f"patch_data/{num_den}/{n_patches}patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            b_fit, m_fit = patch_info["b_fit"], patch_info["m_fit"].flatten()
            amps_rec[i,0], amps_rec[i,1:] = b_fit, m_fit
            xi = patch_info["xi_full"]
            xi_full_patches[i] = xi
        return amps_rec, xi_full_patches
    else:
        print("")