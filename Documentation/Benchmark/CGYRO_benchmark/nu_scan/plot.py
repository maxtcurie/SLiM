import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('output.csv')



plt.clf()
#df_tmp=df[df.modetype=='mtm']
df_tmp=df
#df=df[df.modetype!='mtm']
plt.scatter(df_tmp['NU_EE'],df_tmp['growth_rate'])


plt.xlabel('NU_EE',fontsize=15)
plt.ylabel(r'$\gamma (c_s/a)$',fontsize=15)
#plt.xlim(0.9,4.3)
plt.legend()
plt.show()

df_tmp.to_csv('./other_modes.csv')