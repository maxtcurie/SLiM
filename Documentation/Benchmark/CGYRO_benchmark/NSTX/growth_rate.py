import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('MTM_r0_kappa_scan_output.csv')


marker_size_scale=1000

plt.clf()
df_tmp=df[df.modetype=='mtm']
df=df[df.modetype!='mtm']

plt.scatter([1.42322],[2.2],color='red',marker='*',s=300,label='NSTX')
#plt.scatter([2.877],[1.6333],color='green',marker='*',s=300,label='DIII-D')
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],\
			color='red',label='MTM')
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],\
			s=df_tmp['growth_rate']*marker_size_scale,\
			color='red')

df_tmp=df[df.modetype=='etg']
df=df[df.modetype!='etg']
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],\
			color='blue',label='TEM')
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],\
			s=df_tmp['growth_rate']*marker_size_scale,\
			color='blue')
df_tmp=df[df.modetype=='itg']
df=df[df.modetype!='itg']
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],\
			color='green',label='ITG')
plt.scatter(df_tmp['RMAJ'],df_tmp['KAPPA'],\
			s=df_tmp['growth_rate']*marker_size_scale,\
			color='green')

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