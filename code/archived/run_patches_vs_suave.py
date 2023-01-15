from datetime import datetime
start = datetime.now()

# compute correlation function on lognormal mocks (no gradient!)
import xi_lognormal
xi_lognormal.xi_ls_ln_mocklist()

# generate gradient mocks based on the specified grad_type in globals
import gradmock_gen
gradmock_gen.generate_gradmocks()

# run patches
import patchify_xi
patchify_xi.xi_in_patches()

import patches_lstsq_fit
patches_lstsq_fit.patches_lstsq_fit()

# run suave
import suave
suave.suave_grad(plots=True)

# compare performance of patches and suave
import patches_vs_suave_stats
grads = patches_vs_suave_stats.extract_grads_patches_suave()
grads_exp = grads["grads_exp"]
grads_rec_patches = grads["grads_rec_patches"]
grads_rec_suave = grads["grads_rec_suave"]
# patches_vs_suave_stats.scatter_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave)
patches_vs_suave_stats.histogram_patches_vs_suave(grads_exp, grads_rec_patches, grads_rec_suave, nbins=30)
patches_vs_suave_stats.stats_patches_suave(grads_exp, grads_rec_patches, grads_rec_suave)

runtime = datetime.now() - start
days, seconds = runtime.days, runtime.seconds
hours = days * 24 + seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print(f"total run time = {hours}:{minutes}.{seconds}")