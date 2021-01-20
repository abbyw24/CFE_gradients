import numpy as np
import matplotlib.pyplot as plt

# periodic
periodic = True

# load in xi for split, lognormal, and dead mocks
split_data = np.load("split_xi_per-"+str(periodic)+".npy")
r_avg = split_data[0]
xi_split = split_data[1]
xi_ln = split_data[2]
xi_dead = split_data[3]

# load in vector
v = np.load("split_mock_v.npy")

# average of xi_ln and xi_dead
avg_xi_ln_dead = (xi_dead + xi_ln)/2

fig = plt.figure()
# plot Corrfunc for split
plt.plot(r_avg, xi_split, marker='o', label="Split")
plt.xlabel(r'r ($h^{-1}$Mpc)')
plt.ylabel(r'$\xi$(r)')
plt.title(r"Landy-Szalay, Periodic="+str(periodic))

# plot Corrfunc for lognormal and dead mocks, and the average of the two
plt.plot(r_avg, xi_ln, marker='o', label="Lognormal")
plt.plot(r_avg, xi_dead, marker='o', label="Dead")
plt.plot(r_avg,avg_xi_ln_dead, color="blue",alpha=0.3,marker='o', label="Avg, Lognorm and Dead")
# plot zero line
plt.plot(r_avg, np.zeros(len(r_avg)), color="black",alpha=0.5)

plt.legend()
plt.show()

# save figure
fig.savefig("split_xi_per-"+str(periodic)+".png")