# STEP 1: generate gradient mocks
import grad_mock_gen
# grad_mock_gen takes in a w_hat dimension, array of m and b values, and a lognormal set
#               outputs boxsize metadata, m and b values metadata, mock data set, and both uncolored and color-separated plots of mocks

# STEP 2: run suave on gradient mocks to estimate their correlation functions with Landy-Szalay
import grad_mock_xi
# takes in global boxsize, m and b values, and parameters for suave
# returns array grad_xi of r_avg values and correlation function (shape (nbins,))
#           as well as similar arrays lognormal_xi and dead_xi

# STEP 3: plot correlation function for visualization
import grad_mock_xi_visual
# returns plot of correlation function: r_avg vs. correlation functions for gradient mock, lognormal mock, dead mock,
#                                                   and average of lognormal and dead mocks