import numpy as np
import sys as sys
import csv
sys.path.insert(1, './../Tools')

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive

#************Start of user block******************
#para=   [nu,  zeff,eta, shat, beta,   ky,   mu]
para_min=[0.1, 1.,  0.5, 0.001,0.0005, 0.01, 0.]
para_max=[10.,  5.,  5.,  0.05, 0.02,   0.2,  10.]  

path='.'   
Output_csv=path+'/0MTM_scan_PC.csv'

xstar=10.
ModIndex=1
#************End of user block******************

para_min=np.array(para_min)
para_max=np.array(para_max)
width=(para_max-para_min)

with open(Output_csv, 'w', newline='') as csvfile:     #clear all and then write a row
    csv_data = csv.writer(csvfile, delimiter=',')
    csv_data.writerow(['omega_omega_n','gamma_omega_n',\
            'nu','zeff','eta','shat','beta','ky',\
            'ModIndex','mu','xstar'])
csvfile.close()

while 1==1:
    param=para_min+width*np.random.rand(7)
    [nu,zeff,eta,shat,beta,ky,mu]=param

    w0=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,1,mu,xstar) 
    #w0=1+1j
    omega=np.real(w0)
    gamma=np.imag(w0)
    print(str(omega)+','+str(gamma)+','+str(nu)+','+str(zeff)+','\
                +str(eta)+','+str(shat)+','+str(beta)+','+str(ky)+','\
                +str(ModIndex)+','+str(mu)+','+str(xstar))

    with open(Output_csv, 'a+', newline='') as csvfile: #adding a row
        csv_data = csv.writer(csvfile, delimiter=',')
        csv_data.writerow([ omega,gamma,nu,zeff,eta,shat,beta,ky,\
                ModIndex,mu,xstar ])
    csvfile.close()
    
    print('******w*****')
    print('w='+str(w0))