import gradmock_gen
import xi_funcs_mocks
import bao_iterative
import scipy_fit
import globals
globals.initialize_vals()


compute_baseline = True
compute_cfe = False
compute_patches = False


### baseline scripts ###    
if compute_baseline:
    # # generate gradient mocks
    # print("generate_gradmocks:")
    # gradmock_gen.generate_gradmocks()

    # Landy-Szalay
    print("xi_ls_mocklist:")
    xi_funcs_mocks.xi_ls_mocklist(overwrite=False)


### CFE ###
if compute_cfe:
    # BAO iterative
    # print("bao_iterative:")
    # bao_iterative.main()

    # CFE gradient estimation
    print("grad_cfe_mocklist:")
    xi_funcs_mocks.grad_cfe_mocklist(basis='bao_iterative', overwrite=False)

    # CFE
    print("xi_cfe_mocklist:")
    xi_funcs_mocks.xi_cfe_mocklist(basis='bao_iterative', overwrite=False)


### standard method ###
if compute_patches:
    # 4-parameter fit to get bases
    print("scipy_fit:")
    scipy_fit.main()

    # patches gradient estimation
    print("grad_patches_mocklist:")
    xi_funcs_mocks.grad_patches_mocklist()