import numpy as np
import os
import globals
globals.initialize_vals()

def generate_mock_list(
    boxsize = globals.boxsize,
    lognormal_density = globals.lognormal_density,
    As = globals.As,
    grad_type = globals.grad_type,
    n_mocks = globals.n_mocks,
    extra = False
):
    if As == 2:
        As_key = '_As2x'
    elif As == 1:
        As_key = ''

    lognorm_mock = f'cat_L{boxsize}_n{lognormal_density}_z057_patchy{As_key}'
    path_to_lognorm_source = os.path.join('/scratch/ksf293/mocks/lognormal', lognorm_mock)

    mock_file_name_list = []
    mock_name_list = []

    if grad_type == "1rlz":
        b = 0.5
        m_arr = np.linspace(-1.0, 1.0, n_mocks)
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = [f'{lognorm_mock}_lognormal_rlz1']
        for m in m_arr:
            for b in b_arr:
                mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file_list[0], m, b)
                mock_file_name_list.append(mock_file_name)
                mock_name = "n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m, b)
                mock_name_list.append(mock_name)

    elif grad_type == "1m":
        m = 0.0
        b = 0.5
        m_arr = m * np.ones([n_mocks])
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = []
        for i in range(n_mocks):
            lognorm_file_list.append(f"{lognorm_mock}_lognormal_rlz{i}")

        for lognorm_file in lognorm_file_list:
            mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file, m, b)
            mock_file_name_list.append(mock_file_name)
            mock_name = "n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m, b)
            mock_name_list.append(mock_name)
    
    elif grad_type == "1rlz_per_m":
        b = 0.5
        m_arr = np.linspace(-1.0, 1.0, n_mocks)
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = []
        for i in range(n_mocks):
            lognorm_file_list.append(f"{lognorm_mock}_lognormal_rlz{i}")
            mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file_list[i], m_arr[i], b)
            mock_file_name_list.append(mock_file_name)
            mock_name = "n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m_arr[i], b)
            mock_name_list.append(mock_name)
    
    elif grad_type == "1mock":     # (i.e. plots for poster)
        m = 1.0
        b = 0.5
        m_arr = m * np.ones([n_mocks])
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = [f"{lognorm_mock}_lognormal_rlz400"]
        mock_file_name_list = ["{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file_list[0], m_arr[0], b)]
        mock_name_list = ["n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m_arr[0], b)]

    else:
        print("'grad_type' must be '1rlz', '1m', or '1rlz_per_m'")
        assert False
    
    if extra == True:
        vals = {
            "mock_file_name_list" : mock_file_name_list,
            "mock_name_list" : mock_name_list,
            "lognorm_mock" : lognorm_mock,
            "path_to_lognorm_source" : path_to_lognorm_source
        }
        return vals
    else:
        assert extra == False
        return mock_file_name_list