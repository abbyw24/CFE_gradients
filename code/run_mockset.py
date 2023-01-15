import gradmock_gen
import xi_funcs
import bao_iterative
import scipy_fit

compute_baseline = True
compute_cfe = False
compute_patches = True

### baseline scripts ###    
if compute_baseline:
    # generate gradient mocks
    gradmock_gen.generate_gradmocks()

    # Landy-Szalay
    xi_funcs.xi_ls_mocklist(overwrite=True)


### CFE ###
if compute_cfe:
    # BAO iterative
    bao_iterative.main()

    # CFE gradient estimation
    xi_funcs.grad_cfe_mocklist()

    # CFE
    xi_funcs.xi_cfe_mocklist()


### standard method ###
if compute_patches:
    # 4-parameter fit to get bases
    scipy_fit.main()

    # patches gradient estimation
    xi_funcs.grad_patches_mocklist()