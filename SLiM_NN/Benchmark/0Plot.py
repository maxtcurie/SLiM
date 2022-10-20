import matplotlib.pyplot as plt 
import pandas as pd

SLiM_csv='SLiM_calc.csv'
SLiM_NN_csv='SLiM_calc_NN.csv'

data=pd.read_csv(SLiM_csv)      #data is a dataframe. 
data_NN=pd.read_csv(SLiM_NN_csv)      #data is a dataframe. 


plt.clf()
plt.plot(data['eta'],data['gamma'],label='SLiM')
plt.plot(data_NN['eta'],data_NN['gamma'],label='SLiM_NN')
plt.xlabel(r'$\eta$',fontsize=15)
plt.ylabel(r'$\gamma/\omega_{*n}$',fontsize=15)
plt.axhline(0,color='black',alpha=0.5)
plt.grid(alpha=0.2)
plt.legend()
plt.show()


'''
plt.clf()
plt.plot(data['eta'],data['gamma'],label='SLiM')
plt.plot(data_NN['eta'],data_NN['gamma_10']*max(data['gamma']),label='SLiM_NN')
plt.xlabel(r'$\eta$',fontsize=15)
plt.ylabel(r'$\gamma/\omega_{*n}$',fontsize=15)
plt.axhline(0,color='black',alpha=0.5)
plt.grid(alpha=0.2)
plt.legend()
plt.show()
'''

#data_NN=data_NN.query('gamma_10==1')

plt.clf()
plt.plot(data['eta'],data['f'],label='SLiM')
plt.plot(data_NN['eta'],data_NN['f'],label='SLiM_NN')
plt.xlabel(r'$\eta$',fontsize=15)
plt.ylabel(r'$\omega/\omega_{*n}$',fontsize=15)
plt.grid(alpha=0.2)
plt.legend()
plt.show()

