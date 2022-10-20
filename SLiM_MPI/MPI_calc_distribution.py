# -*- coding: utf-8 -*-
"""
Created on 10/04/2022

@author: maxcurie
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#************Start of user block******************
#para=       [nu,    shat,  beta]
para_min_log=[0.01, 0.001, 0.0005]
para_max_log=[10,   1.,    0.1]  

#para=   [zeff, eta, ky,   mu]
para_min=[1.,   0.5, 0.01, 0.]
para_max=[5.,   5.,  0.3,  10.]

xstar=10.
ModIndex=1

n=100000
#************End of user block******************

para_min_log=np.array(para_min_log)
para_max_log=np.array(para_max_log)
width_log=(para_max_log-para_min_log)

para_min=np.array(para_min)
para_max=np.array(para_max)
width=(para_max-para_min)

para_list=[]

#log_p=1./10**(np.linspace(0.1,1,100000))
#log_p=np.linspace(0.1,1,100000)
#log_p=log_p/np.sum(log_p)

m=10000
j=0
x=np.linspace(para_min_log[j],para_max_log[j], m)
y=np.logspace(np.log10(para_min_log[j]),np.log10(para_max_log[j]), m)

if 1==0:
    plt.clf()
    plt.plot(x,y)
    plt.yscale('log')
    plt.show()

for i in tqdm(range(n)):
    #param_log=para_min_log+width_log*np.random.rand(len(para_min_log))
    #param_log=para_min_log+width_log*( 10.**(1.-np.random.rand(len(para_min_log))) )
    param_log=[]
    for j in range(len(para_min_log)):
        [tmp]=np.random.choice(np.logspace(np.log10(para_min_log[j]),\
                                           np.log10(para_max_log[j]), m), \
                                size=1, p=[1./m]*m)\
        
        param_log.append(tmp)

    param_log=np.array(param_log)
    param=para_min+width*np.random.rand(len(para_min))
    #[nu, shat, beta, zeff, eta, ky, mu]=numpy.concatenate( (param_log,param) )
    para_list.append(np.concatenate( (param_log,param) ))

para_list=np.array(para_list)

name_list=[r'$\nu$', r'$\hat{s}$', r'$\beta$', r'$Z_{eff}$', r'$\eta$', r'$k_y$', r'$\mu$']

fig, ax=plt.subplots(nrows=3,ncols=3,sharex=False) 
            #nrows is the total rows
            #ncols is the total columns
            #sharex true means the xaxies will be shared
print(np.shape(para_list))
for i in range(9):
    if i<3:
        #ax[int(i/3),i%3].hist(para_list[:,i], density=True, bins=80)
        #https://stackoverflow.com/questions/6855710/how-to-have-logarithmic-bins-in-a-python-histogram
        ax[int(i/3),i%3].hist(para_list[:,i], density=False, \
                            bins=np.logspace(np.log10(np.min(para_list[:,i])),\
                                             np.log10(np.max(para_list[:,i])), 80))
        ax[int(i/3),i%3].set_xlabel(name_list[i])
        ax[int(i/3),i%3].set_ylabel('count')
        ax[int(i/3),i%3].set_xscale('log')
    elif i<6:
        #ax[int(i/3),i%3].hist(para_list[:,i], density=False, \
        #                    bins=np.logspace(np.log10(np.min(para_list[:,i])),\
        #                                     np.log10(np.max(para_list[:,i])), 80))
        ax[int(i/3),i%3].hist(para_list[:,i], density=True, bins=80)
        ax[int(i/3),i%3].set_xlabel(name_list[i])
        ax[int(i/3),i%3].set_ylabel('count')
    elif i==6:
        ax[int(i/3),i%3].axis('off')
    elif i==7:
        ax[int(i/3),i%3].hist(para_list[:,6], density=True, bins=80)
        ax[int(i/3),i%3].set_xlabel(name_list[6])
        ax[int(i/3),i%3].set_ylabel('count')
    else:
        ax[int(i/3),i%3].axis('off')

    ax[int(i/3),i%3].set_yticks([])

plt.tight_layout()
plt.show()