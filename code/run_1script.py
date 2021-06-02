import gradmock_gen
gradmock_gen.generate_gradmocks()

# import patches_stats
# n_patches_list = [27, 8]
# lognormal_density = "2e-4"
# patches_stats.histogram_patches(n_patches_list, lognormal_density=lognormal_density, nbins=30)

import density_stats
densities_list = ["4e-4", "2e-4", "1e-4"]
method = "patches"
density_stats.histogram_densities(densities_list, method, nbins=30)