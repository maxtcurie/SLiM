# -*- coding: utf-8 -*-
"""
Created on 10/25/2021

@author: maxcurie
"""

import pandas as pd
import numpy as np
import csv
import time
import random
import concurrent.futures as future #for CPU parallelization
import sys
sys.path.insert(1, './../Tools')

from DispersionRelationDeterminantFullConductivityZeff import Dispersion

#**********Start of user block***************
path='.'   
Output_csv=path+'/0MTM_scan.csv'

nu_list=np.arange(1.,10,1.)
zeff_list=np.arange(1,2.5,0.5)
eta_list=np.arange(0.5,5,0.8)
shat_list=np.arange(0.02,0.1,0.02)
beta_list=np.arange(0.0005,0.003,0.0005)
ky_list=np.arange(0.01,0.1,0.01)
mu_list=np.arange(0,2,0.3)
xstar=10.
ModIndex=1 #global dispersion
#**********end of user block****************

def Dispersion_calc(para_list):
    [nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,Output_csv]=para_list
    
    w0=Dispersion(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar) 

    omega=np.real(w0)
    gamma=np.imag(w0)
    
    with open(Output_csv, 'a+', newline='') as csvfile: #adding a row
        csv_data = csv.writer(csvfile, delimiter=',')
        csv_data.writerow([ omega, gamma,nu,zeff,eta,shat,beta,ky,\
                ModIndex,mu,xstar ])
    csvfile.close()
    return w0

if __name__ == '__main__':      
    with open(Output_csv, 'w', newline='') as csvfile:     #clear all and then write a row
        csv_data = csv.writer(csvfile, delimiter=',')
        csv_data.writerow(['omega_omega_n','gamma_omega_n',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
    csvfile.close()

    para_list=[]
    for nu in nu_list:
        for zeff in zeff_list:
            for eta in eta_list:
                for shat in shat_list:
                    for beta in beta_list:
                        for ky in ky_list:
                            for mu in mu_list:
                                para_list.append([nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,Output_csv])
    print(para_list[0])
    random.shuffle(para_list)
    print(para_list[0])
    print('***********************************')
    print('*********paraellel calcuation******')
    start=time.time()
    with future.ProcessPoolExecutor() as executor:
        results = executor.map(Dispersion_calc, para_list)

    for result0 in results:
        print( 'omega='+str(result0) )

    end=time.time()
    print(f"Runtime of the program is {end - start} s")
    