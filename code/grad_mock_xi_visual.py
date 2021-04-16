import numpy as np
import matplotlib.pyplot as plt
import globals

globals.initialize_vals()

grad_dim = globals.grad_dim
L = globals.L
loop = globals.loop
m_arr_perL = globals.m_arr_perL
b_arr = globals.b_arr

periodic = globals.periodic

# load in xi for ln and dead mocks
ln_xi = np.load("lognormal_xi.npy")
r_avg = ln_xi[0]
xi_ln = ln_xi[1]

dead_xi = np.load("dead_xi.npy")
xi_dead = dead_xi[1]

# average of xi_ln and xi_dead
avg_xi_ln_dead = (xi_ln + xi_dead)/2

# loop through m and b values
for m in m_arr_perL:
    for b in b_arr:
        # define a value in terms of m and b
        a = f"m-{m}-L_b-{b}"
        # load in xi for gradient mock
        grad_data = np.load(f"gradient_mocks/{grad_dim}D/xi/grad_xi_{a}_per-{periodic}.npy")
        xi_grad = grad_data[1]

        fig1 = plt.figure()
        # plot Corrfunc for gradient
        plt.plot(r_avg, xi_grad, marker='o', label=f"Gradient, m={m}/L, b={b}")
        plt.xlabel(r'r ($h^{-1}$Mpc)')
        plt.ylabel(r'$\xi$(r)')
        plt.title(r"Landy-Szalay (Periodic="+str(periodic)+")")

        # plot Corrfunc for lognormal and dead mocks, and the average of the two
        plt.plot(r_avg, xi_ln, marker='o', label="Lognormal")
        plt.plot(r_avg, xi_dead, marker='o', label="Dead")
        plt.plot(r_avg,avg_xi_ln_dead, color="blue",alpha=0.3,marker='o', label="Avg, Lognorm and Dead")
        # plot zero line
        plt.plot(r_avg, np.zeros(len(r_avg)), color="black",alpha=0.5)

        plt.legend()

        # save figure
        fig1.savefig(f"gradient_mocks/{grad_dim}D/xi/grad_xi_{a}_per-{periodic}.png")

        print(f"m={m}, b={b}, done!")