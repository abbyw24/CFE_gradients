import numpy as np

# function to take an array of points and shift them to within specified values
def center_mock(data, min_val, max_val, shift_val=None):
    a = np.all((data >= min_val) & (data <= max_val))
    k = 0
    if shift_val == None:
        shift_val = (max_val - min_val)/2
    while a == False:
        print(f"correction {k}: min patch_center = {data.min()}, max patch_center = {data.max()}")
        if np.any(data <= min_val):
            # print("too low, shifting up by {:.2f}".format(shift_val))
            data += shift_val
        elif np.any(data >= max_val):
            # print("too high, shifting down by {:.2f}".format(shift_val))
            data -= shift_val
        else:
            assert np.all((data >= min_val) & (data <= max_val))
            print(f"mock centered between {min_val} and {max_val}")
            break
        k += 1