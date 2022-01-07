from memory_profiler import profile
from mem_funcs import printsizeof

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import itertools as it
import os
from create_subdirs import create_subdirs
from ls_debug import xi_ls
from center_mock import center_mock_debug   # * change back to center_mock after debugging
import generate_mock_list
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
lognormal_density = globals.lognormal_density
path_to_data_dir = globals.path_to_data_dir
grad_type = globals.grad_type

randmult = globals.randmult
periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_patches = globals.n_patches

mock_file_name_list = generate_mock_list.generate_mock_list()

print("test 1")

# below is different from patchify_debug:
from patchify_debug import patchify

# trial -1a:
#   - "hello world"
#   - import mock_info
#   - center mock
#   - NO create rand_set
#   - NO patchify
#   - NO patch_centers
#   - NO run full xi_ls
#   - NO xi_ls in patches
#   - NO plotting
#   - NO saving info
# RESULT: SUCCESS!!!

# trial -2a:
#   - "hello world"
#   - import mock_info
#   - NO center mock
#   - NO create rand_set
#   - NO patchify
#   - NO patch_centers
#   - NO run full xi_ls
#   - NO xi_ls in patches
#   - NO plotting
#   - NO saving info
# RESULT: SUCCESS!!!

# trial -3:
#   - "hello world"
#   - NO import mock_info
#   - NO center mock
#   - NO create rand_set
#   - NO patchify
#   - NO patch_centers
#   - NO run full xi_ls
#   - NO xi_ls in patches
#   - NO plotting
#   - NO saving info
# RESULT: SUCCESS!!!

# trial -2:
#   - import mock_info
#   - center mock
#   - NO create rand_set
#   - NO patchify
#   - NO patch_centers
#   - NO run full xi_ls
#   - NO xi_ls in patches
#   - NO plotting
#   - NO saving info
# RESULT: NO print outputs at all!

# trial -1:
#   - import mock_info
#   - center mock
#   - create rand_set
#   - NO patchify
#   - NO patch_centers
#   - NO run full xi_ls
#   - NO xi_ls in patches
#   - NO plotting
#   - NO saving info
# RESULT: NO print outputs at all!

# trial 0:
#   - import mock_info
#   - center mock
#   - create rand_set
#   - NO patchify
#   - NO patch_centers
#   - run full xi_ls        *
#   - NO xi_ls in patches
#   - NO plotting
#   - NO saving info
# RESULT: code hangs at mock * (rlz 21 for L750, n2e-4, '1m')

# trial 1:
#   - import mock_info
#   - center mock
#   - create rand_set
#   - patchify mock
#   - patchify rand_set
#   - create patch_centers
#   - run full xi_ls
#   - run xi_ls in each patch
#   - NO plotting
#   - NO saving info
# RESULT: code hangs at mock *

# * bug is definitely within function:
#   calling function in run_1script outputs NO print statements
#   importing patchify_fromscratch but not calling function outputs the print statements outside of function

# vals = np.empty((len(mock_file_name_list), 2))

def xi_in_patches(grad_dim=grad_dim, path_to_data_dir=path_to_data_dir, mock_file_name_list = mock_file_name_list, n_patches=n_patches, n=146):
    print("test 2")

    for i in range(len(mock_file_name_list)):

        # cond = (i >= n)
        # if cond:
        #     print(f"mock {i}:")
        # import mock_info::
        print(f"mock {i}:")
        mock_info = np.load(os.path.join(path_to_data_dir, f"mock_data/{lognormal_density}/{mock_file_name_list[i]}.npy"), allow_pickle=True).item()
        printsizeof(mock_info, on="mock_info")  # cond=cond
        mock_file_name = mock_info["mock_file_name"]
        mock_data = mock_info["grad_set"]
        L = mock_info["boxsize"]

        # vals[i] = np.amin(mock_data), np.amax(mock_data)

        # center mock::
        center_mock_debug(mock_data, 0, L)

        nd = len(mock_data)

        # # create rand_set::
        # nr = randmult*nd
        # rand_set = np.random.uniform(0, L, (nr,3))
        # printsizeof(rand_set, on="rand_set", cond=cond)

        # # patchify mock::
        # patches_mock = patchify(mock_data, L, n_patches=n_patches)
        # patch_ids_mock = patches_mock[0]
        # patch_id_list_mock = np.unique(patch_ids_mock)

        # # patchify rand_set::
        # patches_rand = patchify(rand_set, L, n_patches=n_patches)
        # patch_ids_rand = patches_rand[0]
        # patch_id_list_rand = np.unique(patch_ids_rand)
        # if cond:
        #     print(f"patchify complete for mock {i}")

        # # make sure patch lists match for mock and random, that there's nothing weird going on
        # assert np.all(patch_id_list_mock == patch_id_list_rand)
        # patch_id_list = patch_id_list_mock
        # n_patches = len(patch_id_list)

        # # create patch_centers::
        # patch_centers = []
        # for patch_id in patch_id_list:
        #     patch_data = rand_set[patch_ids_rand == patch_id]
        #     center = np.mean(patch_data, axis=0)
        #     patch_centers.append(center)
        # patch_centers = np.array(patch_centers)

        # whether to print statements from xi_ls
        # prints = True if cond else False

        # # run full xi_ls::
        # if cond:
        #     print(f"computing xi_ls for mock {i}...")
        # results_xi_full = xi_ls(mock_data, rand_set, periodic, nthreads, rmin, rmax, nbins, prints=prints)
        # printsizeof(results_xi_full, on="full xi results", cond=cond)
        # if cond:
        #     print(f"full xi results computed for mock {i}")

        # # run xi_ls in each patch::
        # xi_patches = []
        # for patch_id in patch_id_list:
        #     patch_data = mock_data[patch_ids_mock == patch_id]
        #     patch_rand = rand_set[patch_ids_rand == patch_id]
        #     # # assert False right before script hangs
        #     # if cond and patch_id == 7:
        #     #     assert False
        #     # ###
        #     results_xi_patch = xi_ls(patch_data, patch_rand, periodic, nthreads, rmin, rmax, nbins, prints=prints)
        #     printsizeof(results_xi_patch, on=f"patch {patch_id} xi_ls", cond=cond)
        #     if cond:
        #         print(f"completed xi_ls for patch {patch_id}")
        # xi_patches = np.array(xi_patches)
        # printsizeof(xi_patches, on="xi_patches", cond=cond)

        # print(f"xi in patches --> {mock_file_name}")
    
    # print(vals)