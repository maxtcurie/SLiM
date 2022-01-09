import cython_opt.Dispersion_cython.parity_finder_short_cython as p1
import cython_opt.Dispersion_cython.parity_finder_short as p2
import numpy as np
import time

zgrid=np.arange(-6.28,8,0.00001)
f=np.sin(zgrid)

print('****************************')
print('*********Cython************')
loop_time=10000
start=time.time()
p1(zgrid,f,name=' ',plot=False,report=False)
end=time.time()
print(f"Runtime of the program is {end - start} s")

print('****************************')
print('*Cython with type definition*')
start=time.time()
p2(zgrid,f,name=' ',plot=False,report=False)
end=time.time()
print(f"Runtime of the program is {end - start} s")