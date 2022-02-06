import time 
import numpy as np
import sys
sys.path.insert(1, './..')

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
import Cython_Dispersion
import Cython_Dispersion_0th_order
import Dispersion

run_times=1

nu=0.5
Zeff=1.5
eta=1.5
shat=0.003
beta=0.002
ky=0.01
mu=0.
xstar=10.

print('*******************')
print(' Dispersion')
#original Dispersion
start=time.time()
for i in range(run_times):
    w1=Dispersion.VectorFinder_auto_Extensive(nu,Zeff,eta,shat,beta,ky,1,mu,xstar)
    #w1=(2.566917288361145+0.16496476993107406j)
end=time.time()
print(f"Runtime of the program is {end - start} s")


print('*******************')
print('Cython Dispersion')
start=time.time()
for i in range(run_times):
    #w2=(2.566917288361145+0.16496476993107406j)
    w2=Cython_Dispersion.VectorFinder_auto_Extensive(nu,Zeff,eta,shat,beta,ky,1,mu,xstar)
end=time.time()
print(f"Runtime of the program is {end - start} s")

print('*******************')
print('Cython Dispersion 0th order')
start=time.time()
for i in range(run_times):
    w3=Cython_Dispersion_0th_order.VectorFinder_auto_Extensive(nu,Zeff,eta,shat,beta,ky,1,mu,xstar)
end=time.time()
print(f"Runtime of the program is {end - start} s")


print(w1)
print(w2)
print(w3)
w2=complex(w2)
w3=complex(w3)
print(type(w2))
print(np.abs(w1-w2))
