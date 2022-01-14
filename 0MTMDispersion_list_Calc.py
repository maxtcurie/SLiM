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

from DispersionRelationDeterminantFullConductivityZeff import Dispersion

#**********Start of user block***************
Input_csv='./Output/parameter_list.csv'   
Output_csv='./Output/0MTM_scan.csv'
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

    w0=Dispersion(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar) 
    
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
    if gamma < -0.0005:
        break

end=time.time()
print(f"Runtime of the program is {end - start} s")
