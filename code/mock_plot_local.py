import numpy as np
import matplotlib.pyplot as plt
import read_lognormal

# m and b values
m = 1.00
b = 0.5

# LOGNORMAL SET
Lx, Ly, Lz, N, data = read_lognormal.read("lss_data/lognormal_mocks/cat_L750_n3e-4_lognormal_rlz0.bin")
    # define boxsize based on mock; and N = number of data points
L = Lx
    # L = boxsize

w_hat = np.array([1.0, 0, 0])

x_lognorm, y_lognorm, z_lognorm, vx_lognorm, vy_lognorm, vz_lognorm = data.T
xs_lognorm = (np.array([x_lognorm, y_lognorm, z_lognorm])-(L/2))

# RANDOM SET
# generate a random data set (same size as mock)
xs_rand = np.random.uniform(-L/2,L/2,(3,N))

# INJECT GRADIENT
# for each catalog, make random uniform deviates
rs_clust = np.random.uniform(size=N)
rs_uncl = np.random.uniform(size=N)

# dot product onto the unit vectors
ws_clust = np.dot(w_hat, xs_lognorm)
ws_uncl = np.dot(w_hat, xs_rand)

# threshold
ts_clust_squared = (m/L) * ws_clust + b 
ts_uncl_squared = (m/L) * ws_uncl + b

# assert that ts range from 0 to 1
# assert np.all(ts_clust > 0)
# assert np.all(ts_clust < 1)

# desired indices
I_clust = rs_clust**2 < ts_clust_squared
I_uncl = rs_uncl**2 > ts_uncl_squared

# clustered and unclustered sets for gradient
xs_clust_grad = xs_lognorm.T[I_clust]
xs_unclust_grad = xs_rand.T[I_uncl]

# append to create gradient mock data
xs_grad = np.append(xs_clust_grad, xs_unclust_grad, axis=0)

# plot all points in same color
z_max = -50

xy_slice = xs_grad[np.where(xs_grad[:,2] < z_max)] # select rows where z < z_max

fig1, ax1 = plt.subplots()
plt.plot(xy_slice[:,0], xy_slice[:,1],',')   # plot scatter xy-slice

# plot vector w_hat (no z)
a = 0.35*L     # controls width of w_hat vector in plot
x = np.linspace(-a, a, 10)
# plt.plot(x, (w_hat[1]/w_hat[0])*x, color="green", label=w_hat)
plt.arrow(-a, 0, 2*a, 0, color="black", lw=2, head_width = .1*a, head_length=.2*a, length_includes_head=True, zorder=100)

ax1.set_aspect("equal")      # square aspect ratio
ax1.set_xlabel("x (Mpc/h)")
ax1.set_ylabel("y (Mpc/h)")
ax1.set_title("mock")
plt.show()