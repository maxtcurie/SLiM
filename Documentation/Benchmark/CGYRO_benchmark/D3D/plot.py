import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('output.csv')



plt.clf()
df_tmp=df[df.modetype=='mtm']
df=df[df.modetype!='mtm']

plt.scatter([1.42322],[2.2],color='black',marker='*',s=300,label='NSTX')
plt.scatter([2.877],[1.6333],color='red',marker='*',s=300,label='DIII-D')
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],color='red',label='MTM')

df_tmp=df[df.modetype=='etg']
df=df[df.modetype!='etg']
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],color='blue',label='etg')
df_tmp=df[df.modetype=='itg']
df=df[df.modetype!='itg']
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],color='green',label='itg')

df_tmp=df[df.modetype!='itg']
#plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],color='grey',label='other')

df_tmp=df[df['modetype'].str.contains('_stable', case=False, na=False)]
df=df[df['modetype'].str.contains('_stable', case=False, na=False)==False]
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],color='black',label='stable')

print(df_tmp)

plt.xlabel('R/r',fontsize=15)
plt.ylabel(r'$\kappa$',fontsize=15)
plt.xlim(0.9,4.3)
plt.legend()
plt.show()

df_tmp.to_csv('./other_modes.csv')