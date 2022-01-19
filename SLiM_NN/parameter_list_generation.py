import csv
import numpy as np
from tqdm import tqdm

nu_list=np.arange(0,10,1.)
zeff_list=np.arange(1,2.5,0.5)
eta_list=np.arange(0.5,5,0.8)
shat_list=np.arange(0,0.1,0.02)
beta_list=np.arange(0.0001,0.003,0.0005)
ky_list=np.arange(0.01,0.1,0.01)
mu_list=np.arange(0,2,0.3)
xstar=10.
Output_csv='./parameter_list.csv'   

with open(Output_csv, 'w', newline='') as csvfile:     #clear all and then write a row
    csv_data = csv.writer(csvfile, delimiter=',')
    csv_data.writerow(['n','m','rho_tor',\
                'omega_plasma_kHz','omega_lab_kHz',\
                'gamma_cs_a','omega_n_kHz',\
                'omega_n_cs_a','omega_e_plasma_kHz',\
                'omega_e_lab_kHz','peak_percentage',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
total_len=len(nu_list)*len(zeff_list)*len(eta_list)*len(shat_list)*len(beta_list)*len(ky_list)*len(mu_list)
print('total dispersion entry:' + str(total_len))
print('run time(hr):' + str(float(total_len)/2./60.))
print('run time(days):' + str(float(total_len)/2./60./24.))


for nu in tqdm(nu_list):
    for zeff in zeff_list:
        for eta in eta_list:
            for shat in shat_list:
                for beta in beta_list:
                    for ky in ky_list:
                        for mu in mu_list:
                            with open(Output_csv, 'a+', newline='') as csvfile: #adding a row
                                csv_data = csv.writer(csvfile, delimiter=',')
                                csv_data.writerow([ 0,0,0,0,0,0,0,0,0,0,0,\
                                    nu,zeff,eta,shat,beta,ky,1,mu,xstar])
