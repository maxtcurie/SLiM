import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('output.csv')

fontsize0=12


fig, ax=plt.subplots(nrows=2,ncols=1,sharex=True) 
df['n']=df['ky']/0.034282055

convert_dict={}
convert_dict['n']='int'

df=df.astype(convert_dict)

df=df[df.n%2==0]
df_itg=df[df.modetype=='itg']
df_mtm=df[df.modetype=='mtm']
df_tem=df[df.modetype=='etg']
df_kbm=df[df.modetype=='mystery']


df_stable=df[df['modetype'].str.contains('_stable', case=False, na=False)]


ax[0].scatter(df_mtm['ky'],df_mtm['growth_rate'],label='MTM',color='red')
ax[0].scatter(df_itg['ky'],df_itg['growth_rate'],color='blue')
#ax[0].scatter(df_tem['ky'],df_tem['growth_rate'],label='TEM',color='blue')
#ax[0].scatter(df_kbm['ky'],df_kbm['growth_rate'],color='blue')

#ax[0].scatter(df_stable['ky'],[0]*len(df_stable),label='stable',color='black')


ax[1].scatter(df_itg['ky'],df_itg['frequency']*15.14720546-df_itg['ky']/0.034282055*1.714409104,color='blue')
ax[1].scatter(df_mtm['ky'],df_mtm['frequency']*15.14720546-df_mtm['ky']/0.034282055*1.714409104,color='red')
ax[1].scatter(df_tem['ky'],df_tem['frequency']*15.14720546-df_tem['ky']/0.034282055*1.714409104,color='blue')
ax[1].scatter(df_kbm['ky'],df_kbm['frequency']*15.14720546-df_kbm['ky']/0.034282055*1.714409104,color='blue')
ax[1].scatter(df_stable['ky'],[0]*len(df_stable),label='stable',color='black')


ax[0].set_ylabel(r'$\gamma (c_s/a)$',fontsize=fontsize0)
ax[1].set_ylabel(r'$frequency (kHz)$',fontsize=fontsize0)
ax[1].set_xlabel(r'$k_y\rho_s$',fontsize=fontsize0)
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()