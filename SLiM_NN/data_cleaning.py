# -*- coding: utf-8 -*-
"""
Created on 10/25/2021

@author: maxcurie
"""

import pandas as pd
import numpy as np
import csv

from data_tool import load_data
from data_tool import find_data_range
#****************************************
#**********start of user block***********
filename_list=['./NN_data/0MTM_scan_CORI_2.csv',
                './NN_data/0MTM_scan_CORI_1.csv',
                './NN_data/0MTM_scan_CORI_3_large_nu.csv',
                './NN_data/0MTM_scan_CORI_np_rand_V2.csv',
                './NN_data/0MTM_scan_CORI_np_rand_V3_1.csv',
                './NN_data/0MTM_scan_CORI_np_rand_V3_2.csv',
                './NN_data/0MTM_scan_PC_np_rand_V3_2022_10_23.csv',
                './NN_data/0MTM_scan_PC_np_rand_V3_2022_10_23_2.csv']

#****************************************
#**********start of user block***********


df=load_data(filename_list)

df_unstable=df.query('omega_omega_n!=0 and gamma_omega_n>0')
df_stable=df.query('omega_omega_n==0 or gamma_omega_n<=0')


print('len(df_unstable)='+str(len(df_unstable)))
print('len(df_stable)='+str(len(df_stable)))
print('total='+str(len(df)))

for key in df.keys():
    print(key+'='+str(np.min(df[key]))+'~'+str(np.max(df[key])))


nu=1.398957621
zeff=2.788549247
eta=1.158925479
shat=0.005906846
beta=0.000710695
ky=0.040731854
mu=0.1
xstar=10.

para_list=np.array([nu, zeff, eta, shat,  beta,  ky,   mu, xstar],dtype='float')
para_list_min=para_list-para_list*1.
para_list_max=para_list+para_list*1.
print(para_list_min)
print(para_list_max)

df_stable=df.query(find_data_range(para_list_min,para_list_max))

print(df_stable)