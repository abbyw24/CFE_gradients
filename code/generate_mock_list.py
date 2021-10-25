import numpy as np
import os
import globals
globals.initialize_vals()

def generate_mock_list(
    boxsize = globals.boxsize,
    lognormal_density = globals.lognormal_density,
    As = globals.As,
    cat_tag = globals.cat_tag,
    mock_type = globals.mock_type,
    n_mocks = globals.n_mocks,
    m = globals.m,
    b = globals.b,
    rlz = globals.rlz,
    extra = False
):

    lognorm_mock = f'cat_{cat_tag}'
    path_to_lognorm_source = os.path.join('/scratch/ksf293/mocks/lognormal', lognorm_mock)

    mock_file_name_list = []
    mock_name_list = []

    if mock_type == "1rlz":
        m_arr = np.linspace(-1.0, 1.0, n_mocks)
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = [f'{lognorm_mock}_lognormal_rlz{rlz}']
        for m in m_arr:
            for b in b_arr:
                mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file_list[0], m, b)
                mock_file_name_list.append(mock_file_name)
                mock_name = "n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m, b)
                mock_name_list.append(mock_name)

    elif mock_type == "1m":
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
    
    elif mock_type == "1rlz_per_m":
        assert n_mocks == 1 or n_mocks ==41 or n_mocks == 401, "'n_mocks' must be 1, 41, or 401"
        m_arr = np.linspace(-1.0, 1.0, n_mocks)
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = []
        for i in range(n_mocks):
            lognorm_file_list.append(f"{lognorm_mock}_lognormal_rlz{i}")
            mock_file_name = "{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file_list[i], m_arr[i], b)
            mock_file_name_list.append(mock_file_name)
            mock_name = "n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m_arr[i], b)
            mock_name_list.append(mock_name)
    
    elif mock_type == "1mock":     # (i.e. plots for poster)
        assert n_mocks == 1, "'n_mocks' must be 1"
        m_arr = m * np.ones([n_mocks])
        b_arr = b * np.ones([n_mocks])
        lognorm_file_list = [f"{lognorm_mock}_lognormal_rlz{rlz}"]
        mock_file_name_list = ["{}_m-{:.3f}-L_b-{:.3f}".format(lognorm_file_list[0], m_arr[0], b)]
        mock_name_list = ["n{}, m={:.3f}, b={:.3f}".format(lognormal_density, m_arr[0], b)]
    
    elif mock_type == "lognormal":
        m_arr = None
        b_arr = None
        mock_name_list = None
        lognorm_file_list = []
        for i in range(n_mocks):
            filename = f"{lognorm_mock}_lognormal_rlz{i}"
            lognorm_file_list.append(filename)
            mock_file_name_list.append(filename)

    else:
        print("'mock_type' must be '1rlz', '1m', or '1rlz_per_m'")
        assert False
    
    if extra == True:
        vals = {
            "mock_file_name_list" : mock_file_name_list,
            "mock_name_list" : mock_name_list,
            "lognorm_mock" : lognorm_mock,
            "lognorm_file_list" : lognorm_file_list,
            "path_to_lognorm_source" : path_to_lognorm_source,
            "m_arr" : m_arr,
            "b_arr" : b_arr
        }
        return vals
    else:
        assert extra == False
        return mock_file_name_list