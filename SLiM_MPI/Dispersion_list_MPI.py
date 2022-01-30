# -*- coding: utf-8 -*-
"""
Created on 01/26/2022

@author: maxcurie
"""

import numpy as np
import pandas as pd
import csv
from mpi4py import MPI

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
from MPI_tools import task_dis

#**********Start of user block***************
path='.'   
Output_csv=path+'/0Disperson_calc.csv'
Input_csv=path+'/parameter_list.csv'
log_file_name=path+'/Disperson_calc_W_auto.log'
#**********end of user block****************


comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()


print('*******rank='+str(rank)+'*************')

if rank==0:
    with open(Output_csv, 'w', newline='') as csvfile:     #clear all and then write a row
        csv_data = csv.writer(csvfile, delimiter=',')
        csv_data.writerow(['omega_omega_n','gamma_omega_n',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
    csvfile.close()

    with open(log_file_name, 'w') as csvfile:        #clear all and then write a row
        data = csv.writer(csvfile, delimiter=',')
        data.writerow(['init_guess','omega','gamma',\
            'w0','ratio','nu','Zeff','eta','shat',\
            'beta','ky','ModIndex','mu','xstar'])
    csvfile.close()

    df=pd.read_csv(Input_csv)
    para_list=[]
    for i in range(len(df['n'])):
        nu=float(df['nu'][i])
        zeff=float(df['zeff'][i])
        eta=float(df['eta'][i])
        shat=float(df['shat'][i])
        beta=float(df['beta'][i])
        ky=float(df['ky'][i])
        ModIndex=int(df['ModIndex'][i])
        mu=float(df['mu'][i])
        xstar=float(df['xstar'][i])
        para_list.append([nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,Output_csv])

    task_list = task_dis(size,para_list)

    for i in range(size-1):
        comm.send(task_list[i],dest=i+1) #sending the data

elif rank!=0:
    task_list_rank=comm.recv(source=0)  #recieve the data
    for para in task_list_rank:
        [nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,Output_csv]=para
    
        w0=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,file_name=log_file_name) 
        #w0=0.+0j
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
        
        print(para)
            