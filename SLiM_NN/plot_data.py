# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#****************************************
#**********start of user block***********
filename='./NN_data/0MTM_scan_CORI_2.csv'
plot=0 #plot 0 for plotting all
#**********end of user block*************
#****************************************

df=pd.read_csv(filename)
df_unstable=df.query('omega_omega_n!=0 and gamma_omega_n>0 ')
df_stable=df.query('omega_omega_n!=0 and gamma_omega_n<=0 ')
print(len(df_unstable))
print(len(df_stable))
x=np.arange(0,np.max(df_unstable['omega_omega_n'])*1.5,0.01)

if plot==1 or plot==0:
    plt.clf()
    plt.scatter(1.+df_unstable['eta'],df_unstable['omega_omega_n'],label='data')
    plt.plot(x,1.4*x,color='orange',label='y=1.4*x')
    plt.plot(x,x,color='red',label='y=x')
    plt.plot(x,x-0.8,color='green',label='y=x-0.8')
    plt.title(r'relation between frequency and $\eta$')
    plt.xlabel(r'1+$\eta$')
    plt.ylabel(r'$\omega/\omega_n$')
    plt.legend()
    plt.grid()
    plt.show()

    plt.clf()
    plt.scatter(np.arange(len(df_unstable['omega_omega_n']/(1.+df_unstable['eta']))),df_unstable['omega_omega_n']/(1.+df_unstable['eta']),label='data')
    plt.ylabel(r'$\omega/\omega_{*e}$')
    plt.grid()
    plt.show()

if plot==2 or plot==0:
    plt.clf()
    plt.scatter(df_unstable['nu'],df_unstable['gamma_omega_n'],label='data')
    #plt.plot(x,x,color='orange',label='y=x')
    plt.title(r'relation between growth rate and $\nu$')
    plt.xlabel(r'$\nu_{ei}/\omega_n$')
    plt.ylabel(r'$\gamma/\omega_n$')
    #plt.grid()
    plt.legend()
    plt.show()

if plot==3 or plot==0:
    plt.clf()
    plt.scatter(df['omega_omega_n'],df['gamma_omega_n'],label='data')
    plt.plot(x,0.05+x**2.*0.012,color='orange',label=r'y=0.012*$x^2$+0.05')
    plt.title(r'relation between growth rate and frequency')
    plt.xlabel(r'$\omega/\omega_n$')
    plt.ylabel(r'$\gamma/\omega_n$')
    plt.ylim(0,0.8)
    plt.grid()
    plt.legend()
    plt.show()