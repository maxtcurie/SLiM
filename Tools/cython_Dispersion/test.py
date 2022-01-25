import time 
import numpy as np

from A_maker import A_maker
from A_maker2 import A_maker2
import Cython_A_maker_V2

run_times=30

x_max=20.
del_x=0.02
w1=2.5+0.5j
v1=0.5 #nu
Zeff=2.
eta=15.
alpha=0.005 #shat
beta=0.0001
ky=0.01
ModIndex=1
mu=0.01
xstar=10.

print('A_maker--cleaned up')
start=time.time()
for i in range(run_times):
    A_maker2(x_max, del_x, w1, v1,Zeff,eta,alpha,beta,ky,ModIndex,mu,xstar)
end=time.time()
print(f"Runtime of the program is {end - start} s")

print('*******************')
print('A_maker')
start=time.time()
for i in range(run_times):
    A_maker(x_max, del_x, w1, v1,Zeff,eta,alpha,beta,ky,ModIndex,mu,xstar)
end=time.time()
print(f"Runtime of the program is {end - start} s")


print('*******************')
print('Cython A_maker')
start=time.time()
for i in range(run_times):
    Cython_A_maker_V2.A_maker(x_max, del_x, w1, v1,Zeff,eta,alpha,beta,ky,ModIndex,mu,xstar)
end=time.time()
print(f"Runtime of the program is {end - start} s")

