import numpy as np
import os
import globals
globals.initialize_vals()



class MockSet:

    def __init__(self, boxsize, lognormal_density, As=2, data_dir=globals.data_dir, rlzs=globals.rlzs):

        As_tag = '' if As==1 else f'_As{As}x'
        self.cat_tag = f'L{boxsize}_n{lognormal_density}_z057_patchy{As_tag}'
        self.data_dir = data_dir

        # if realizations are specified, use those; otherwise, use the number of mocks set in globals and start from rlz0
        self.rlzs = np.arange(rlzs) if type(rlzs)==int else rlzs
        self.nmocks = len(self.rlzs)

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
    

    ## LOAD TOOLS

    def load_rlz(self, rlz):
        mock_dict = np.load(os.path.join(self.data_dir, f'catalogs/{self.mock_type}/{self.cat_tag}/{self.mock_fn_list[rlz]}.npy'), allow_pickle=True).item()
        return mock_dict
    

    def load_xi_lss(self, nbins=globals.nbins, randmult=globals.randmult):
        ls_dir = os.path.join(self.data_dir, f'{self.mock_path}/ls/{self.cat_tag}')
        xi_lss = np.empty((self.nmocks, nbins))
        for i in range(self.nmocks):
            r, xi_lss[i] = np.load(os.path.join(ls_dir, f'xi_ls_{randmult}x_{self.mock_fn_list[i]}.npy'), allow_pickle=True)
        self.r_avg = r
        self.xi_lss = xi_lss 
    

    def load_xi_cfes(self, bao_fixed=True, ncont=globals.ncont):
        # which BAO basis to use
        basis_type = 'bao_fixed' if bao_fixed else 'bao_iterative'

        cfe_dir = os.path.join(self.data_dir, f'{self.mock_path}/suave/xi/{basis_type}/{self.cat_tag}')
        xi_cfes = np.empty((self.nmocks, ncont))
        for i in range(self.nmocks):
            rcont, xi_cfes[i] = np.load(os.path.join(cfe_dir, f'xi_{self.mock_fn_list[i]}.npy'), allow_pickle=True).T
        self.rcont = rcont
        self.xi_cfes = xi_cfes