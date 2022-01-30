# -*- coding: utf-8 -*-
"""
Created on 01/26/2022

@author: maxcurie
"""

import numpy as np
import csv
from mpi4py import MPI

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
from MPI_tools import task_dis

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print('*******rank='+str(rank)+'*************')

if rank==0:
    #**********Start of user block***************
    path='.'   
    Output_csv=path+'/0MTM_scan.csv'
    
    nu_list=np.arange(0.1,10.,0.5)
    zeff_list=np.arange(1,2.5,0.2)
    eta_list=np.arange(0.5,3.,0.2)
    shat_list=np.arange(0.02,0.1,0.01)
    beta_list=np.arange(0.0005,0.003,0.0003)
    ky_list=np.arange(0.01,0.1,0.01)
    mu_list=np.arange(0,4.,0.1)
    xstar=10.
    ModIndex=1 #global dispersion
    #**********end of user block****************

    with open(Output_csv, 'w', newline='') as csvfile:     #clear all and then write a row
        csv_data = csv.writer(csvfile, delimiter=',')
        csv_data.writerow(['omega_omega_n','gamma_omega_n',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
    csvfile.close()

    with open('./W_auto.log', 'w') as csvfile:        #clear all and then write a row
        data = csv.writer(csvfile, delimiter=',')
        data.writerow(['init_guess','omega','gamma',\
            'w0','ratio','nu','Zeff','eta','shat',\
            'beta','ky','ModIndex','mu','xstar'])
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
    task_list = task_dis(size,para_list)

    for i in range(size-1):
        comm.send(task_list[i],dest=i+1) #sending the data

elif rank!=0:
    task_list_rank=comm.recv(source=0)  #recieve the data
    for para in task_list_rank:
        [nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,Output_csv]=para
    
        w0=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar) 
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
            