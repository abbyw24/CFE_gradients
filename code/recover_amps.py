import numpy as np
import os

def recover_amps(mock_file_name_list, num_den, n_patches, method="suave"):
    amps_rec = np.empty((len(mock_file_name_list), 4))
    abs_path = "/scratch/aew492/research-summer2020_output/1D/"
    if method == "suave":
        for i in range(len(mock_file_name_list)):
            suave_info = np.load(os.path.join(abs_path, f"suave_data/1e-4/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            amps = suave_info["amps"]
            amps_rec[i] = amps
    elif method == "patches":
        for i in range(len(mock_file_name_list)):
            patch_info = np.load(os.path.join(abs_path, f"patch_data/1e-4/8patches/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
            b_fit, m_fit = patch_info["b_fit"], patch_info["m_fit"].flatten()
            amps_rec[i,0], amps_rec[i,1:] = b_fit, m_fit
    else:
        print("")
    return amps_rec