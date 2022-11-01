import gradmock_gen
import xi_funcs

# L = 1500
# m = 1

# L = 1000
# m = 0.667

# L = 750
# m = 0.5

L = 500
m = 0.333

### baseline scripts ###    

# generate gradient mocks
gradmock_gen.generate_gradmocks(L=L, m=m)

# Landy-Szalay
xi_funcs.xi_ls_mocklist(L=L, m=m)

### CFE ###

# CFE gradient estimation
xi_funcs.grad_cfe_mocklist(L=L, m=m)

# CFE
xi_funcs.xi_cfe_mocklist(L=L, m=m)


### standard method ###

# ...
