import numpy as np
import matplotlib.pyplot as plt

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

n_patches = n_sides**3

# create plot
fig1 = plt.figure()
plt.xlabel("Recovered Gradient")

# line at x = 0
plt.vlines(0, 0, 1, color="black", alpha=0.5)

# create an array of recovered gradient values for y and z
grads_recovered_s = []
grads_recovered_p = []

for m in m_arr_perL:
    for b in b_arr:
        suave_data = np.load(f"gradient_mocks/{grad_dim}D/suave/exp_vs_rec_vals/suave_exp_vs_rec_vals_m-{m}-L_b-{b}.npy", allow_pickle=True).item()
        grads_recovered_s.append(suave_data["grad_recovered"])

        patches_data = np.load(f"gradient_mocks/{grad_dim}D/patches/lst_sq_fit/patches_exp_vs_rec_vals_m-{m}-L_b-{b}_{n_patches}patches.npy", allow_pickle=True).item()
        grads_recovered_p.append(patches_data["grad_recovered"])

print(grads_recovered_s.shape)
print(grads_recovered_p.shape)