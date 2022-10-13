import numpy as np

# function to take an array of points and shift them to within specified values
def center_mock(data, min_val, max_val, shift_val=None):
    k = 0
    if shift_val == None:
        shift_val = (max_val - min_val)/2
    while np.all((data >= min_val) & (data <= max_val)) == False:
        # print(f"correction {k}: min data = {data.min()}, max data = {data.max()}")
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
        if k >= 10:
            print("error centering mock– too many iterations")
            print(f"min. data = {np.amin(data)}, max. data = {np.amax(data)}")
            assert False