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

# # plot with quiver
# fig1 = plt.figure()
# V = np.vstack([grad_expected,grad_recovered])
# origin = np.array([[0,0],[0,0]])
# scale = 0.003
# width = 0.005

# plt.quiver(*origin, V[:,0], V[:,1], color=["r","g"], alpha=0.8, width=width, scale=scale)

# # projection of recovered onto expected
# plt.quiver(*origin, proj_rec_onto_exp[0], proj_rec_onto_exp[1], color="b", alpha=0.4, width=width, scale=scale)

# plot "standard way"
V = np.array([grad_expected,grad_recovered,proj_rec_onto_exp])
colors = ["red", "black", "light_blue"]
plot_array = np.column_stack((V, colors))
for i in range(len(V)):
    plt.plot([0,V[i,0]], [0,V[i,1]], color=V[i,3], alpha=0.7, linewidth=2)
# plot residual as triangle
plt.plot([proj_rec_onto_exp[0],grad_recovered[0]],[proj_rec_onto_exp[1],residual[1]], color="gray", linewidth=2)

plt.title("Expected vs. Recovered Clustering Gradient: \n m="+str(m)+", b="+str(b)+", "+str(n_patches)+" patches")
plt.xlabel("x")
plt.ylabel("y")
    # what are the units of this measurement???
plt.legend()
    # can't figure out how to get a legend with quiver
plt.show()

fig1.savefig("gradient_mocks/"+str(grad_dim)+"D/patches/lst_sq_fit/grad_exp_vs_rec_m-"+str(m)+"-L_b-"+str(b)+"_"+str(n_patches)+"patches.png")