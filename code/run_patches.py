import numpy as np

# STEP 1: divide mock into patches and compute xi across all bins
import patchify_xi
# patchify_xi returns the patch centers and xi data for the least square fit,
#   and a plot of the correlation function for each patch

# STEP 2: least square fit of clustering amplitudes in each patch in each bin;
#           m_fit for x, y, z directions, and b_fit, is the avg of the fit value up to a certain bin cutoff;
#               currently, bin_cutoff = int(nbins/2)
#               reuse nbins from patchify_xi
L = np.load("boxsize.npy")
import patches_lst_sq_loop  # this does a least square fit of the clustering amplitudes
                            #   in each patch for each bin
# patches_lst_sq_loop returns the recovered fit values: m_fit_x, m_fit_y, m_fit_z, b_fit,
#   and a plot of the correlation function for each patch, and the recovered values across all bins

# STEP 3: compare recovered vs. expected gradient
import patches_gradient
# patches_gradient returns the recovered gradient (m_fit/b_fit),
#   and the expected gradient (m_input/b_input)w_hat,
#   and a visualisation of the recovered vs. expected and residual over a plot of the mock data