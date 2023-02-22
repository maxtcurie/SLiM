# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from file_list import file_list

#****************************************
#**********start of user block***********
filename_list=file_list()
plot=7 #plot 0 for plotting all
#**********end of user block*************
#****************************************
count=0
for i,filename in zip(range(len(filename_list)),filename_list):
    if 'rand' not in filename:
        continue 
    df=pd.read_csv(filename)
    df=df.dropna()
    try:
        df=df.drop(columns=['change'])
    except:
        pass
    
    if count==0:
        df_merge=df
    else:
        df_merge=pd.concat([df_merge, df], axis=0)
    count=count+1

df=df_merge
df_unstable=df.query('omega_omega_n!=0 and gamma_omega_n>0 ')
df_stable=df.query('omega_omega_n!=0 or gamma_omega_n<=0 ')
print('****************')
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

if plot==4 or plot==0:
    name_list=['nu','zeff','eta','shat','beta','ky','mu']
    fig, ax=plt.subplots(nrows=1,ncols=len(name_list))

    for i in range(len(name_list)):
        #https://stackoverflow.com/questions/33203645/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data
        ax[i].hist(df[name_list[i]], density=False, bins=30)  # density=False would make counts
        ax[i].set_xlabel(name_list[i])
        if i!=0:
            ax[i].set_yticklabels([])
        else:
            ax[i].set_ylabel('Count')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()


if plot==5 or plot==0:
    name_list=['nu','zeff','eta','shat','beta','ky','mu']
    fig, ax=plt.subplots(nrows=1,ncols=len(name_list))

    for i in range(len(name_list)):
        ax[i].scatter(df_unstable[name_list[i]],df_unstable['gamma_omega_n'],alpha=0.2)  # density=False would make counts
        #ax[i].plot(df_unstable[name_list[i]],df_unstable['gamma_omega_n'],alpha=0.2)  # density=False would make counts
        ax[i].set_xlabel(name_list[i])
        ax[i].set_ylabel('gamma/omega_n')
        if i!=0:
            ax[i].set_yticklabels([])
        else:
            ax[i].set_ylabel('Density')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()



if plot==6 or plot==0:
    plt.clf()
    plt.scatter(df_unstable['nu'],df_unstable['gamma_omega_n'],alpha=0.2)  # density=False would make counts
    #ax[i].plot(df_unstable['nu'],df_unstable['gamma_omega_n'],alpha=0.2)  # density=False would make counts
    plt.xlim(0,5)
    plt.xlabel('nu')
    plt.ylabel('gamma/omega_n')  
    plt.show()


if plot==7 or plot==0:
    alpha_=0.1
    s_=0.1
    plt.clf()
    df_=df_unstable
    plt.scatter(df_['nu']/df_['eta'],df_['beta']/df_['shat']**2,s=s_,color='red',alpha=alpha_)  # density=False would make counts
    alpha_=0.01
    df_=df_stable
    #print(df_)
    plt.scatter(df_['nu']/df_['eta'],df_['beta']/df_['shat']**2,s=s_,color='blue',alpha=alpha_)  # density=False would make counts
    plt.xlabel(r'$\nu/\omega_{*e}$')
    plt.ylabel(r'$\beta/\hat{s}^2$')  
    plt.xscale('log')
    plt.yscale('log')
    plt.show()