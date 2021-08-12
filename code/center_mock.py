import numpy as np

# function to take an array of points and shift them to within specified values
def center_mock(data, min_val, max_val, shift_val=None):
    k = 0
    if shift_val == None:
        shift_val = (max_val - min_val)/2
    while np.all((data >= min_val) & (data <= max_val)) == False:
        # print(f"correction {k}: min patch_center = {data.min()}, max patch_center = {data.max()}")
        if np.any(data <= min_val):
            # print("too low, shifting up by {:.2f}".format(shift_val))
            data += shift_val
        elif np.any(data >= max_val):
            # print("too high, shifting down by {:.2f}".format(shift_val))
            data -= shift_val
        else:
            assert np.all((data >= min_val) & (data <= max_val))
            break
        k += 1
        print("centering mock...")

def center_mock_debug(data, min_val, max_val, shift_val=None):
    print("center_mock:")
    print(f"max value = {np.amax(data)}, min value = {np.amin(data)}")
    k = 0
    if shift_val == None:
        shift_val = (max_val - min_val)/2
    while np.all((data >= min_val) & (data <= max_val)) == False:
        # print(f"correction {k}: min patch_center = {data.min()}, max patch_center = {data.max()}")
        if np.any(data <= min_val):
            print("too low, shifting up by {:.2f}".format(shift_val))
            data += shift_val
            print(f"new max value = {np.amax(data)}, new min = {np.amin(data)}")
            print(f"a = {np.all((data >= min_val) & (data <= max_val))}")
        elif np.any(data >= max_val):
            print("too high, shifting down by {:.2f}".format(shift_val))
            data -= shift_val
            print(f"new max value = {np.amax(data)}, new min = {np.amin(data)}")
            print(f"a = {np.all((data >= min_val) & (data <= max_val))}")
        else:
            assert np.all((data >= min_val) & (data <= max_val))
            break
        k += 1
        # if k >= 2:
        #     break