import matplotlib.pyplot as plt

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

# create plot
fig1 = plt.figure()
plt.xlabel("Expected Gradient")
plt.ylabel("Recovered Gradient")

expected_xgrads = []

# loop through m and b values
for m in m_arr_perL:
    for b in b_arr:
        # load in data from suave gradient recovery
        suave_data = np.load(f"gradient_mocks/{grad_dim}D/suave/suave_exp_vs_rec_vals_m-{m}-L_b-{b}.npy")
        m_s, b_s, amps, grad_expected_s, grad_recovered_s, mean_sq_err_s = suave_data
        
        # load in data from patches gradient recovery
        patches_data = np.load(f"gradient_mocks/{grad_dim}D/patches/exp_vs_rec_vals/patches_exp_vs_rec_vals_m-{m}-L_b-{b}_{n_patches}patches.npy")
        m_p, b_p, n_patches, grad_expected_p, grad_recovered_p, mean_sq_err_p = patches_data

        # make sure the data that should match does actually match
        assert m_s == m_p
        m = m_s
        assert b_s == b_p
        b = b_s
        assert grad_expected_s == grad_expected_p
        grad_expected = grad_expected_s
        # save expected gradients so we can pull out the max value for plotting
        expected_xgrads.append(grad_expected[0])

        # plot expected vs. recovered (in x direction only) for suave
        plt.plot(grad_expected[0], grad_recovered_s[0], marker="o", color="blue")

        # plot expected vs. recovered (in x direction only) for suave
        plt.plot(grad_expected[0], grad_recovered_p[0], marker="o", color="green")

# plot line y = x (the data points would fall on this line if the expected and recovered gradients matched up perfectly)
x = np.linspace(0, max(expected_xgrads), 10)
plt.plot(x, x, color="black", alpha=0.5)

plt.show()

fig1.savefig(f"gradient_mocks/{grad_dim}D/exp_vs_rec/exp_vs_rec.png")