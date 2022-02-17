import numpy as np
import os
import globals
globals.initialize_vals()

def generate_mock_list(
    cat_tag = globals.cat_tag,
    mock_type = globals.mock_type,
    nmocks = globals.nmocks,
    m = globals.m,
    b = globals.b,
    rlz = globals.rlz,
    extra = False
):

    lognorm_mock = f'cat_{cat_tag}'
    path_to_lognorm_source = os.path.join('/scratch/ksf293/mocks/lognormal', lognorm_mock)

    mock_file_name_list = []

    if mock_type == "1rlz":

        m_arr = np.linspace(-1.0, 1.0, nmocks)
        b_arr = b * np.ones([nmocks])
        lognorm_file_list = [f'{cat_tag}_lognormal_rlz{rlz}']

        for i in range(nmocks):
            mock_param_list[i] = "m-{:.3f}-L_b-{:.3f}".format(m_arr[i], b_arr[i])
            mock_file_name_list[i] = "{}_rlz{}_{}".format(cat_tag, rlz, mock_param_list[i])


    elif mock_type == "1m":

        m_arr = m * np.ones([nmocks])
        b_arr = b * np.ones([nmocks])
        lognorm_file_list = []
        mock_param_list = []

        for i in range(nmocks):
            lognorm_file_list.append(f"{cat_tag}_lognormal_rlz{i}")
            mock_param_list.append("m-{:.3f}-L_b-{:.3f}".format(m, b))
            mock_file_name_list.append("{}_rlz{}_{}".format(cat_tag, i, mock_param_list[i]))

    
    elif mock_type == "1rlz_per_m":

        assert nmocks == 1 or nmocks == 41 or nmocks == 401, "'nmocks' must be 1, 41, or 401"
        m_arr = np.linspace(-1.0, 1.0, nmocks)
        b_arr = b * np.ones([nmocks])
        lognorm_file_list = []
        mock_param_list = []

        for i in range(nmocks):
            lognorm_file_list.append(f"{cat_tag}_lognormal_rlz{i}")
            mock_param_list.append("m-{:.3f}-L_b-{:.3f}".format(m_arr[i], b_arr[i]))
            mock_file_name_list.append("{}_rlz{}_{}".format(cat_tag, i, mock_param_list[i]))

    
    elif mock_type == "1mock":     # (i.e. plots for poster)
        assert nmocks == 1, "'nmocks' must be 1"
        m_arr = m * np.ones([nmocks])
        b_arr = b * np.ones([nmocks])
        lognorm_file_list = [f"{cat_tag}_lognormal_rlz{rlz}"]
        mock_param_list = ["m-{:.3f}-L_b-{:.3f}".format(m, b)]
        mock_file_name_list = ["{}_rlz{}_{}".format(cat_tag, rlz, mock_param_list[0])]
    
    elif mock_type == "lognormal":
        m_arr = None
        b_arr = None
        mock_param_list = None
        lognorm_file_list = []

        for i in range(nmocks):
            filename = f"{cat_tag}_lognormal_rlz{i}"
            lognorm_file_list.append(filename)
            mock_file_name_list.append("{}_rlz{}_lognormal".format(cat_tag, i))

    else:
        print("'mock_type' must be '1rlz', '1m', or '1rlz_per_m'")
        assert False
    
    if extra == True:
        vals = {
            "mock_file_name_list" : mock_file_name_list,
            "lognorm_mock" : lognorm_mock,
            "lognorm_file_list" : lognorm_file_list,
            "path_to_lognorm_source" : path_to_lognorm_source,
            "m_arr" : m_arr,
            "b_arr" : b_arr,
            "mock_param_list" : mock_param_list
        }
        return vals
    else:
        assert extra == False
        return mock_file_name_list