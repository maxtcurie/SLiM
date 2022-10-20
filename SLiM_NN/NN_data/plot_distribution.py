# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#****************************************
#**********start of user block***********
filename_list=['./NN_data/0MTM_scan_CORI_2.csv',
                './NN_data/0MTM_scan_PC.csv',
                './NN_data/0MTM_scan_CORI_1.csv',
                './NN_data/0MTM_scan_CORI_3_large_nu.csv']
plot=5 #plot 0 for plotting all
#**********end of user block*************
#****************************************

for i,filename in zip(range(len(filename_list)),filename_list):
    df=pd.read_csv(filename)
    df=df.dropna()
    try:
        df=df.drop(columns=['change'])
    except:
        pass

    if i==0:
        df_merge=df
    elif i!=0:
        df_merge=pd.concat([df_merge, df], axis=0)

df=df_merge
df_unstable=df.query('omega_omega_n!=0 and gamma_omega_n>0 ')
df_stable=df.query('omega_omega_n!=0 or gamma_omega_n<=0 ')
print(len(df_unstable))
print(len(df_stable))
x=np.arange(0,np.max(df_unstable['omega_omega_n'])*1.5,0.01)