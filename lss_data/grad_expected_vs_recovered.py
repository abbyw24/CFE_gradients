import numpy as np
import matplotlib.pyplot as plt

# ** this is currently for the x dimension only **

######
# load in patch data
m = 0.5
b = 0.5
grad_dim = 1
n_patches = 8
nbins = 8
L = np.load("boxsize.npy")
######

# load in recovered gradient
recovered_vals = np.load("gradient_mocks/"+str(grad_dim)+"D/patches/lst_sq_fit/recovered_vals_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.npy")
m_x_rec, m_y_rec, m_z_rec, b_rec = recovered_vals
grad_recovered = (1/b_rec)*np.array([m_x_rec,m_y_rec,m_z_rec])
print("recovered gradient =", grad_recovered)

# expected gradient (only in x direction)
grad_expected = np.array([m/(b*L),0,0])
print("expected gradient =", grad_expected)

# projection of recovered onto expected
grad_exp_norm = np.sqrt(sum(grad_expected**2))
proj_rec_onto_exp = (np.dot(grad_recovered,grad_expected)/grad_exp_norm**2)*grad_expected

# residual
residual = grad_recovered - proj_rec_onto_exp

# load in mock data
xs_clust_grad = np.load("gradient_mocks/"+str(grad_dim)+"D/mocks_colored/clust_m-"+str(m)+"-L_b-"+str(b)+".npy")
xs_uncl_grad = np.load("gradient_mocks/"+str(grad_dim)+"D/mocks_colored/unclust_m-"+str(m)+"-L_b-"+str(b)+".npy")

z_max = 100

xy_slice_clust = xs_clust_grad[np.where(xs_clust_grad[:,2] < z_max)]
xy_slice_uncl = xs_uncl_grad[np.where(xs_uncl_grad[:,2] < z_max)]

fig1 = plt.figure()
plt.plot(xy_slice_clust[:,0],xy_slice_clust[:,1],',',c="C0")
plt.plot(xy_slice_uncl[:,0],xy_slice_uncl[:,1],',',c="orange")

# plot expected, recovered, and projection from origin (only in xy)
V = np.array([proj_rec_onto_exp, grad_expected, grad_recovered])
colors = np.array(["blue", "red", "black"])
labels = np.array([r"Proj. of $\vec{r}$ onto $\vec{e}$", r"Expected ($\vec{e}$)", r"Recovered ($\vec{r}$)"])
plot_array = np.column_stack((V, colors, labels))
a = 200*L     # scale factor

for i in range(len(V)):
    plt.plot([0,a*V[i,0]], [0,a*V[i,1]], label=plot_array[i,4], color=plot_array[i,3], alpha=0.8, linewidth=2)
# plot residual as triangle
plt.plot([a*proj_rec_onto_exp[0],a*grad_recovered[0]], [a*proj_rec_onto_exp[1],a*residual[1]], color="black", alpha=0.4, linewidth=2)

plt.title("Expected vs. Recovered Clustering Gradient: \n m="+str(m)+", b="+str(b)+", "+str(n_patches)+" patches")
plt.axes().set_aspect("equal")      # square aspect ratio
plt.ylim((-400,400))
plt.xlim((-400,400))
plt.xlabel("x (Mpc/h)")
plt.ylabel("y (Mpc/h)")
plt.legend()
plt.show()

fig1.savefig("gradient_mocks/"+str(grad_dim)+"D/patches/lst_sq_fit/grad_exp_vs_rec_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.png")