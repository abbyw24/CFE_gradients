import multiprocessing as mp
from itertools import repeat
from time import sleep


# arguments
a = 1
b = [2,3,4]

def func(a=1, b=2, d=4):
    c = a+b
    print(c)

with mp.Pool() as pool:
    mp_res = pool.starmap(func, zip(repeat(a), b))
    mp_res

# shut down mp properly
sleep(1)
pool.close()
pool.join()
sleep(1)