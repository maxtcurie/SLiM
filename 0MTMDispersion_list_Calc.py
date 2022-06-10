# -*- coding: utf-8 -*-
"""
Created on 10/25/2021

@author: maxcurie
"""

import pandas as pd
import numpy as np
import csv
import sys
import time
sys.path.insert(1, './Tools')

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
from SLiM_NN.Dispersion_NN import Dispersion_NN

#**********Start of user block***************
Input_csv='./Test_files/parameter_list.csv'   
Output_csv='./Output/0MTM_scan.csv'

Run_mode=3      #Run_mode=1 quick calculation(30sec/mode)
                #Run_mode=2 extensive(30min/mode)
                #Run_mode=3  NN mode (global) (0.05sec/mode)

#**********end of user block****************
    
df=pd.read_csv(Input_csv)
with open(Output_csv, 'w', newline='') as csvfile:     #clear all and then write a row
    csv_data = csv.writer(csvfile, delimiter=',')
    csv_data.writerow(['n','m','rho_tor',\
                'omega_plasma_kHz','omega_lab_kHz',\
                'gamma_cs_a','omega_n_kHz',\
                'omega_n_cs_a','omega_e_plasma_kHz',\
                'omega_e_lab_kHz','peak_percentage',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
csvfile.close()

start=time.time()

if Run_mode==3:
    from SLiM_NN.Dispersion_NN import Dispersion_NN
    
    path='./SLiM_NN/Trained_model/'
    NN_omega_file      =path+'SLiM_NN_omega.h5'
    NN_gamma_file      =path+'SLiM_NN_stabel_unstable.h5'
    norm_omega_csv_file=path+'NN_omega_norm_factor.csv'
    norm_gamma_csv_file=path+'NN_stabel_unstable_norm_factor.csv'

    Dispersion_NN_obj=Dispersion_NN(NN_omega_file,NN_gamma_file,norm_omega_csv_file,norm_gamma_csv_file)


for i in range(len(df['n'])):
    nu=df['nu'][i]
    zeff=df['zeff'][i]
    eta=df['eta'][i]
    shat=df['shat'][i]
    beta=df['beta'][i]
    ky=df['ky'][i]
    ModIndex=df['ModIndex'][i]
    mu=df['mu'][i]
    xstar=df['xstar'][i]

    if Run_mode==1:
        w0=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar) 
    elif Run_mode==2:
        w0=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar) 
    elif Run_mode==3:
        w0=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar) 

    omega=np.real(w0)
    omega_kHz=omega*df['omega_n_kHz'][i]
    gamma=np.imag(w0)
    gamma_cs_a=gamma*df['omega_n_cs_a'][i]

    with open(Output_csv, 'a+', newline='') as csvfile: #adding a row
        csv_data = csv.writer(csvfile, delimiter=',')
        csv_data.writerow([ df['n'][i],df['m'][i],df['rho_tor'][i],\
                    omega_kHz,omega_kHz+df['omega_e_lab_kHz'][i]-df['omega_e_plasma_kHz'][i],gamma_cs_a,\
                    df['omega_n_kHz'][i],df['omega_n_cs_a'][i],\
                    df['omega_e_plasma_kHz'][i],df['omega_e_lab_kHz'][i],
                    df['peak_percentage'][i],df['nu'][i],\
                    df['zeff'][i],df['eta'][i],\
                    df['shat'][i],df['beta'][i],df['ky'][i],\
                    df['ModIndex'][i],df['mu'][i],df['xstar'][i] ])
    csvfile.close()

end=time.time()
print(f"Runtime of the program is {end - start} s")
