# compare performance of patches and suave
import patches_vs_suave_stats
grads = patches_vs_suave_stats.extract_grads_patches_suave()
grads_exp = grads["grads_exp"]
grads_rec_patches = grads["grads_rec_patches"]
grads_rec_suave = grads["grads_rec_suave"]
patches_vs_suave_stats.scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave)
patches_vs_suave_stats.histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, nbins=20)
patches_vs_suave_stats.stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave)

import patches_stats
n_patches_list = [27, 8]
lognormal_density = "2e-4"
patches_stats.histogram_patches(n_patches_list, lognormal_density=lognormal_density, nbins=30)