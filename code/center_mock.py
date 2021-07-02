import numpy as np

def center_mock(patch_centers, min_val, max_val, shift_val=None):
    a = np.all((patch_centers >= min_val) & (patch_centers <= max_val))
    k = 0
    if shift_val == None:
        shift_val = (max_val - min_val)/2
    while a == False:
        print(f"correction {k}: min patch_center = {patch_centers.min()}, max patch_center = {patch_centers.max()}")
        if np.any(patch_centers <= min_val):
            print("too low, shifting up by {:.2f}".format(shift_val))
            patch_centers += shift_val
        elif np.any(patch_centers >= max_val):
            print("too high, shifting down by {:.2f}".format(shift_val))
            patch_centers -= shift_val
        else:
            assert np.all((patch_centers >= min_val) & (patch_centers <= max_val))
            break
        k += 1