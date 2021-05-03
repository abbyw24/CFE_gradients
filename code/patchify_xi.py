import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import Corrfunc
import itertools as it
import globals

globals.initialize_vals()  # brings in all the default parameters

grad_dim = globals.grad_dim
L = globals.L
loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

periodic = globals.periodic
rmin = globals.rmin
rmax = globals.rmax
nbins = globals.nbins
nthreads = globals.nthreads

n_sides = globals.n_sides

# define patchify
def patchify(data, boxsize, n_sides=2):
    nd, n_dim = data.shape
    boxsize_patch = boxsize/n_sides
    a = np.arange(n_sides)
    idx_patches = np.array(list(it.product(a, repeat=n_dim)))
    patch_ids = np.zeros(nd).astype(int)
    for patch_id,iii in enumerate(idx_patches):
        # define the boundaries of the patch
        mins = iii*boxsize_patch
        maxes = (iii+1)*boxsize_patch
        # define mask as where all of the values must be within the boundaries
        mask_min = np.array([(data[:,d]<maxes[d]) for d in range(n_dim)])
        mask_max = np.array([(data[:,d]>=mins[d]) for d in range(n_dim)])
        mask = np.vstack([mask_min, mask_max]).T
        mask_combined = np.all(mask, axis=1)
        # perform masking and save to array
        patch_ids[mask_combined] = patch_id
    return patch_ids, idx_patches

# define Corrfunc Landy-Szalay
def xi(data, rand_set, periodic=False, rmin=20.0, rmax=100.0, nbins=22):
    # parameters
    r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    nr = len(rand_set)

    x, y, z = data[:,0], data[:,1], data[:,2]
    x_rand, y_rand, z_rand = rand_set[:,0], rand_set[:,1], rand_set[:,2]

    dd_res = Corrfunc.theory.DD(1, nthreads, r_edges, x, y, z, periodic=periodic)
    dr_res = Corrfunc.theory.DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, periodic=periodic)
    rr_res = Corrfunc.theory.DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=periodic)

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd,nd,nr,nr,dd,dr,dr,rr)

    return r_avg, results_xi

# loop through the m and b values
for m in m_arr_perL:
    for b in b_arr:
        mock_data = np.load(f"gradient_mocks/{grad_dim}D/mocks/grad_mock_m-{m}-L_b-{b}.npy")
            # x,y,z values from -L/2 to L/2
        # shift values to 0 to L
        mock_data += L/2
        nd = len(mock_data)

        # create random set
        nr = 2*nd
        rand_set = np.random.uniform(0, L, (nr,3))

        # patchify mock data and random set
        patches_mock = patchify(mock_data, L, n_sides=n_sides)
        patch_ids_mock = patches_mock[0]
        patch_id_list_mock = np.unique(patch_ids_mock)

        patches_rand = patchify(rand_set, L, n_sides=n_sides)
        patch_ids_rand = patches_rand[0]
        patch_id_list_rand = np.unique(patch_ids_rand)

        # make sure patch lists match for mock and random, that there's nothing weird going on
        assert np.all(patch_id_list_mock == patch_id_list_rand)
        patch_id_list = patch_id_list_mock
        n_patches = len(patch_id_list)
        patches_idx = patches_mock[1]

        # define patch centers by taking mean of the random set in each patch
        patch_centers = []
        for patch_id in patch_id_list:
            patch_data = rand_set[patch_ids_rand == patch_id]
            center = np.mean(patch_data, axis=0)
            patch_centers.append(center)
        patch_centers = np.array(patch_centers)
        np.save(f"gradient_mocks/{grad_dim}D/patches/patch_centers/patch_centers_m-{m}-L_b-{b}_{n_patches}patches.npy", patch_centers)

        # results for full mock
        results_xi_full = xi(mock_data, rand_set)
        xi_full = np.array(results_xi_full[1])

        # define r_avg (this is the same for all xi)
        r_avg = results_xi_full[0]

        # results in patches
        xi_patches = []
        k = 0
        fig = plt.figure()
        ax = plt.axes()
        cmap = plt.cm.get_cmap("cool")
        ax.set_prop_cycle('color', cmap(np.linspace(0, 1, n_patches)))

        for patch_id in patch_id_list:
            patch_data = mock_data[patch_ids_mock == patch_id]
            patch_rand = rand_set[patch_ids_rand == patch_id]
            results_xi_patch = xi(patch_data, patch_rand)
            xi_patch = results_xi_patch[1]
            #print(f"m={m}, b={b}, patch {k+1} done")

            plt.plot(r_avg, xi_patch, alpha=0.5, marker=".", label=patches_idx[k])
            xi_patches.append(xi_patch)
            k += 1
        xi_patches = np.array(xi_patches)

        # average of patch results
        xi_patch_avg = np.sum(xi_patches, axis=0)/len(xi_patches)

        # save xi data– to load in separate file for least square fit
        patches_xi = {
            "r_avg" : r_avg,
            "xi_patches" : xi_patches,
            "xi_patch_avg" : xi_patch_avg,
            "xi_full" : xi_full
            }
        np.save(f"gradient_mocks/{grad_dim}D/patches/grad_xi_m-{m}-L_b-{b}_{n_patches}patches.npy", patches_xi, allow_pickle=True)

        # plot results
        plt.plot(r_avg, xi_full, color="black", marker=".", label="Full Mock")
        plt.plot(r_avg, xi_patch_avg, color="black", alpha=0.5, marker=".", label="Avg. of Patches")
        # plot parameters
        plt.xlabel(r'r ($h^{-1}$Mpc)')
        plt.ylabel(r'$\xi$(r)')
        plt.rcParams["axes.titlesize"] = 10
        plt.title(f"Standard Estimator, Grad Mock Patches, {grad_dim}D, m={m}/L, b={b}")
        plt.legend(prop={'size': 8})
        fig.savefig(f"gradient_mocks/{grad_dim}D/patches/grad_xi_m-{m}-L_b-{b}_{n_patches}patches.png")

if loop == False:
    # pull up color mock for reference
    fig2 = plt.figure()
    ax = fig2.add_axes([0, 0, 1, 1])
    ax.axis('off')
    im = img.imread(f"gradient_mocks/{grad_dim}D/mocks_colored/color_grad_mock_m-{m}-L_b-{b}.png")
    ax.imshow(im)
    plt.show()