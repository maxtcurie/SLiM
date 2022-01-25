import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#**********************************************
#**********Start of User block*****************

file_list =['./../Discharge_survey/D3D/169510/1966/global_dispersion.csv',\
            './../Discharge_survey/D3D/169510/3069/global_dispersion.csv',\
            './../Discharge_survey/D3D/169510/4069/global_dispersion.csv']

name_list=['t=1966ms','t=3090ms','t=4069ms']

nmin=1
nmax=5
#************End of User Block*****************
#**********************************************

fig, ax=plt.subplots(nrows=2,\
        ncols=len(name_list),sharex=True,sharey='row') 
for i in range(len(file_list)):
    file_name=file_list[i]
    df=pd.read_csv(file_name)
    df_unstabel=df.query('gamma_cs_a>0')
    df_unstabel=df_unstabel[((nmin<=df_unstabel.n) & (df_unstabel.n<=nmax))]

    df_stabel  =df.query('gamma_cs_a<=0')
    df_stabel  =df_stabel[((nmin<=df_stabel.n) & (df_stabel.n<=nmax))]

    n_stable=np.unique(np.array(df_stabel['n']))
    for n in df_unstabel['n']:
        if n in n_stable:
            index=np.argmin(abs(n_stable-n))
            n_stable=np.delete(n_stable,index)

    ax[0,i].scatter(df_unstabel['n'],df_unstabel['gamma_cs_a'],color='red',label='unstable MTM')
    ax[1,i].scatter(df_unstabel['n'],df_unstabel['omega_lab_kHz'],color='red',label='unstable MTM')
    ax[0,i].scatter(n_stable,[0]*len(n_stable),marker='x',color='black',label='stable MTM')
    ax[1,i].scatter(n_stable,[0]*len(n_stable),marker='x',color='black',label='stable MTM')

ax[0,len(file_list)-1].legend()

(nr,nc)=np.shape(ax)
for i in range(nr):
    for j in range(nc):
        ax[i,j].grid()
        if i==nr-1:
            ax[i,j].set_xlabel('toroidal mode nubmer')
        if i==0:
            ax[i,j].set_title(name_list[j])

    ax[0,0].set_ylabel(r'$\gamma(c_s/a)$')
    ax[1,0].set_ylabel('Frequency(kHz)')

plt.subplots_adjust(wspace=0, hspace=0.03)
plt.show()
