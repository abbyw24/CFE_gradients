import numpy as np
import matplotlib.pyplot as plt

# load in m and b lists for loop
m_list_noL = np.load("m_values_noL.npy")
b_list = np.load("b_values.npy")

######
# define dimension
dim = "1D"
######

# define periodic
periodic = False

# load in xi for split, ln, and dead mocks
split_data = np.load("split_xi_per-"+str(periodic)+".npy")
r_avg = split_data[0]
xi_split = split_data[1]
xi_ln = split_data[2]
xi_dead = split_data[3]

# average of xi_ln and xi_dead
avg_xi_ln_dead = (xi_dead + xi_ln)/2

for m in m_list_noL:
    for b in b_list:
        # define a value in terms of m and b
        a = "m-"+str(m)+"-L_b-"+str(b)
        # load in xi for gradient mock
        grad_data = np.load("gradient_mocks/"+dim+"/xi/grad_xi_"+a+"_per-"+str(periodic)+".npy")
        xi_grad = grad_data[1]

        fig1 = plt.figure()
        # plot Corrfunc for gradient
        plt.plot(r_avg, xi_grad, marker='o', label="Gradient, m="+str(m)+"/L, b="+str(b))
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
        fig1.savefig("gradient_mocks/"+dim+"/xi/grad_xi_"+a+"_per-"+str(periodic)+".png")

        print("m="+str(m)+", b="+str(b)+", done!")