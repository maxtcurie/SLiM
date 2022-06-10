import matplotlib.pyplot as plt 
import pandas as pd

SLiM_csv='SLiM_calc_2.csv'
SLiM_NN_csv='SLiM_calc_NN.csv'

data=pd.read_csv(SLiM_csv)      #data is a dataframe. 
data_NN=pd.read_csv(SLiM_NN_csv)      #data is a dataframe. 

data=data.query('gamma>0.00001 and nu<14')

plt.clf()
plt.plot(data['nu'],data['gamma'],label='SLiM')
plt.plot(data_NN['nu'],data_NN['gamma_10']*max(data['gamma']),label='SLiM_NN')
plt.xlabel(r'$\nu_{ei}/\omega_{*n}$',fontsize=15)
plt.ylabel(r'$\gamma/\omega_{*n}$',fontsize=15)
plt.axhline(0,color='black',alpha=0.5)
plt.grid(alpha=0.2)
plt.xlim(0,20)
plt.legend()
plt.show()

data_NN2=data_NN.query('gamma_10==1 and nu<14 ')


plt.clf()
plt.plot(data['nu'],data['f'],label='SLiM')
plt.plot(data_NN2['nu'],data_NN2['f'],label='SLiM_NN')
plt.xlabel(r'$\nu_{ei}/\omega_{*n}$',fontsize=15)
plt.ylabel(r'$\omega/\omega_{*n}$',fontsize=15)
plt.ylim(0, 1.2*max(max(data_NN2['f']),max(data['f'])) )
plt.grid(alpha=0.2)
plt.xlim(0,17)
plt.legend()
plt.show()

fig, ax=plt.subplots(nrows=2,ncols=1,sharex=True) 
			#nrows is the total rows
			#ncols is the total columns
			#sharex true means the xaxies will be shared

ax[0].plot(data['nu'],data['gamma'],label='SLiM')
ax[0].plot(data_NN['nu'],data_NN['gamma_10']*max(data['gamma']),label='SLiM_NN')
ax[0].axhline(0,color='black',alpha=0.5)
ax[0].set_ylabel(r'$\gamma/\omega_{*n}$',fontsize=15)
ax[0].legend()
ax[0].grid(alpha=0.2)

ax[1].plot(data['nu'],data['f'],label='SLiM')
ax[1].plot(data_NN2['nu'],data_NN2['f']*data_NN2['gamma_10'],label='SLiM_NN')
ax[1].set_xlabel(r'$\nu_{ei}/\omega_{*n}$',fontsize=15)
ax[1].set_ylabel(r'$\omega/\omega_{*n}$',fontsize=15)
ax[1].set_ylim(0, 1.2*max(max(data_NN2['f']),max(data['f'])) )
ax[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()