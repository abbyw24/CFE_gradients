# # generate gradient mocks based on the specified grad_type in globals
# import gradmock_gen
# gradmock_gen.generate_gradmocks()

# # run patches
# import patchify_xi
# patchify_xi.xi_in_patches()

import patches_lstsq_fit
patches_lstsq_fit.patches_lstsq_allbins()
patches_lstsq_fit.patches_lstsq_fit_1bin(r_bin=2)

# run suave
import suave
suave.suave_exp_vs_rec()

# compare performance of patches and suave
import patches_vs_suave_stats
grads = patches_vs_suave_stats.extract_grads_patches_suave()
grads_exp = grads["grads_exp"]
grads_rec_patches = grads["grads_rec_patches"]
grads_rec_suave = grads["grads_rec_suave"]
patches_vs_suave_stats.scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave)
patches_vs_suave_stats.histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, nbins=20)
patches_vs_suave_stats.stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave)