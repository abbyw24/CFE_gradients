import numpy as np
import os
import globals
globals.initialize_vals()



class mock_set:

    def __init__(self, boxsize, lognormal_density, As=2, data_dir=globals.data_dir, rlzs=None, nmocks=globals.nmocks):

        self.cat_tag = f'L{boxsize}_n{lognormal_density}_z057_patchy_As{As}x'
        self.data_dir = data_dir

        # if realizations are specified, use those; otherwise, use the number of mocks set in globals and start from rlz0
        if rlzs is not None:
            self.rlzs = np.arange(rlzs) if type(rlzs)==int else rlzs
            self.nmocks = len(rlzs)
        else:
            self.rlzs = range(nmocks)
            self.nmocks = nmocks

        self.ln_fn_list = [f'{self.cat_tag}_rlz{rlz}_lognormal' for rlz in self.rlzs]

        # these will change if we call the add_gradient() method
        self.mock_type = 'lognormal'
        self.mock_fn_list = self.ln_fn_list
        self.mock_path = 'lognormal'


    def add_gradient(self, grad_dim, m, b):
        self.grad_dim = grad_dim
        self.m = m
        self.b = b
        self.grad_dir = os.path.join(self.data_dir, f'gradient/{self.grad_dim}D/{self.cat_tag}')

        self.mock_type = 'gradient'      # redefine the mock type from 'lognormal'
        self.mock_path = f'gradient/{grad_dim}D'    # redefine the mock path (extra layer of specifying gradient dimension)

        # create arrays of our gradient parameters m and b:
        #   if the input m is a single number, this will be distributed across each realization;
        #   otherwise, the list of ms and bs will be multiplied element-wise (in this case len(m)==len(b)==nmocks)
        m_arr = np.multiply(m, np.ones(self.nmocks))
        b_arr = np.multiply(b, np.ones(self.nmocks))
        self.m_arr = m_arr
        self.b_arr = b_arr
        # self.mock_param_list = np.column_stack((m_arr, b_arr))

        # generate a list of the gradient mock file names
        mock_fn_list = []
        for i in range(self.nmocks):
            mock_fn_list.append(f'{self.cat_tag}_rlz{self.rlzs[i]}_m-{m_arr[i]:.3f}-L_b-{b_arr[i]:.3f}')
        self.mock_fn_list = mock_fn_list





























# def generate_mock_list(
#     cat_tag = globals.cat_tag,
#     mock_type = globals.mock_type,
#     nmocks = globals.nmocks,
#     m = globals.m,
#     b = globals.b,
#     rlz = globals.rlz,
#     extra = False
# ):

#     lognorm_mock = f'cat_{cat_tag}'
#     path_to_lognorm_source = os.path.join('/scratch/ksf293/mocks/lognormal', lognorm_mock)

#     mock_file_name_list = []
#     mock_param_list = []
#     lognorm_file_list = []
#     rlz_list = []

#     if mock_type == "1rlz":

#         m_arr = np.linspace(-1.0, 1.0, nmocks)
#         b_arr = b * np.ones(nmocks)

#         for i in range(nmocks):
#             lognorm_file_list.append(f'{cat_tag}_rlz{rlz}_lognormal')
#             mock_param_list.append("m-{:.3f}-L_b-{:.3f}".format(m_arr[i], b_arr[i]))
#             mock_file_name_list.append("{}_rlz{}_{}".format(cat_tag, rlz, mock_param_list[i]))
#             rlz_list.append(rlz)


#     elif mock_type == "1m":

#         m_arr = m * np.ones(nmocks)
#         b_arr = b * np.ones(nmocks)

#         for i in range(nmocks):
#             lognorm_file_list.append(f"{cat_tag}_rlz{i}_lognormal")
#             mock_param_list.append("m-{:.3f}-L_b-{:.3f}".format(m, b))
#             mock_file_name_list.append("{}_rlz{}_{}".format(cat_tag, i, mock_param_list[i]))
#             rlz_list.append(i)

    
#     elif mock_type == "1rlz_per_m":

#         assert nmocks == 1 or nmocks == 41 or nmocks == 401, "'nmocks' must be 1, 41, or 401"
#         m_arr = np.linspace(-1.0, 1.0, nmocks)
#         b_arr = b * np.ones(nmocks)

#         for i in range(nmocks):
#             lognorm_file_list.append(f"{cat_tag}_rlz{i}_lognormal")
#             mock_param_list.append("m-{:.3f}-L_b-{:.3f}".format(m_arr[i], b_arr[i]))
#             mock_file_name_list.append("{}_rlz{}_{}".format(cat_tag, i, mock_param_list[i]))
#             rlz_list.append(i)

    
#     elif mock_type == "1mock":     # (i.e. plots for poster)
#         assert nmocks == 1, "'nmocks' must be 1"
#         m_arr = m * np.ones(nmocks)
#         b_arr = b * np.ones(nmocks)
#         lognorm_file_list.append(f"{cat_tag}_rlz{rlz}_lognormal")
#         mock_param_list.append("m-{:.3f}-L_b-{:.3f}".format(m, b))
#         mock_file_name_list.append("{}_rlz{}_{}".format(cat_tag, rlz, mock_param_list[0]))
#         rlz_list.append(rlz)
    
#     elif mock_type == "lognormal":
#         m_arr = None
#         b_arr = None
#         mock_param_list = None

#         for i in range(nmocks):
#             filename = f"{cat_tag}_rlz{i}_lognormal"
#             lognorm_file_list.append(filename)
#             mock_file_name_list.append("{}_rlz{}_lognormal".format(cat_tag, i))
#             rlz_list.append(i)

#     else:
#         print("'mock_type' must be '1rlz', '1m', or '1rlz_per_m'")
#         assert False
    
#     if extra == True:
#         vals = {
#             "mock_file_name_list" : mock_file_name_list,
#             "lognorm_mock" : lognorm_mock,
#             "lognorm_file_list" : lognorm_file_list,
#             "path_to_lognorm_source" : path_to_lognorm_source,
#             "m_arr" : m_arr,
#             "b_arr" : b_arr,
#             "mock_param_list" : mock_param_list,
#             "rlz_list" : rlz_list
#         }
#         return vals
#     else:
#         assert extra == False
#         return mock_file_name_list