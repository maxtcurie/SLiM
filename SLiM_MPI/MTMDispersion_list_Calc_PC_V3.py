# -*- coding: utf-8 -*-
"""
Created on 01/26/2022

@author: maxcurie
"""

import numpy as np
import csv

from Dispersion import VectorFinder_auto_Extensive


#make sure the csv exist and have the first line:
#omega_omega_n,gamma_omega_n,nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar

#************Start of user block******************
#para=       [nu,    shat,  beta,   ]
para_min_log=[0.01, 0.001, 0.0005 ]
para_max_log=[10,   1.,    0.1]  

#para=   [zeff, eta, ky  ]
para_min=[1.,   0.5, 0.01]
para_max=[5.,   5.,  0.3 ]


path='./'   
Output_csv=path+'0MTM_scan_PC_np_rand.csv'

xstar=10.
ModIndex=1
#************End of user block******************
para_min_log=np.array(para_min_log)
para_max_log=np.array(para_max_log)
width_log=(para_max_log-para_min_log)

para_min=np.array(para_min)
para_max=np.array(para_max)
width=(para_max-para_min)

m=10000

while 1==1:
    param_log=[]
    for j in range(len(para_min_log)):
        [tmp]=np.random.choice(np.logspace(np.log10(para_min_log[j]),\
                                           np.log10(para_max_log[j]), m), \
                                size=1, p=[1./m]*m)\
        
        param_log.append(tmp)

    param_log=np.array(param_log)
    param=para_min+width*np.random.rand(len(para_min))

    [nu, shat, beta, zeff, eta, ky]=np.concatenate( (param_log,param) )

    mu=0.
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

