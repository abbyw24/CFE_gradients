import numpy as np

# import patches_lstsq_fit
# patches_lstsq_fit.patches_lstsq_allbins()
# patches_lstsq_fit.patches_lstsq_fit_1bin(r_bin=2)

# import patches_stats
# n_patches_list = [27, 8]
# lognormal_density = "2e-4"
# patches_stats.histogram_patches(n_patches_list, lognormal_density=lognormal_density, nbins=30)

# import density_stats
# densities_list = ["4e-4", "2e-4", "1e-4"]
# method = "CFE"
# density_stats.histogram_densities(densities_list, method, nbins=30)

patch_info = np.load("/scratch/aew492/research-summer2020_output/1Dpatch_data/2e-4/8patches/cat_L500_n2e-4_z057_patchy_As2x_lognormal_rlz0_m--1.000-L_b-0.500.npy", allow_pickle=True).item()
print(patch_info)
print(patch_info["grad_recovered"])

import patches_vs_suave_stats
grads = patches_vs_suave_stats.extract_grads_patches_suave()
grads_exp = grads["grads_exp"]
grads_rec_patches = grads["grads_rec_patches"]
grads_rec_suave = grads["grads_rec_suave"]
patches_vs_suave_stats.scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave)
patches_vs_suave_stats.histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, nbins=30)
patches_vs_suave_stats.stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave)