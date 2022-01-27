# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:58:16 2020
modified on 10/26/2021
@author: jlara, maxcurie
"""

import numpy as np
from matplotlib import pyplot as plt
from max_parity_calculator import parity_finder_short
import csv

#The following code solves the electromagnetic slab model equations.
#It has been developed to study that dispersion characteristics of micro-tearing modes a single branch or solution
#of these equations. The code will find other branches of solution such as twisting modes and higher excited modes. This may be of interest to some users.

#The code requires some user input such as dimensionless parameters (described below)
#To run the code, call the VectorFinder function with the following arguments.
#nu: the electron-electron collision frequency normalized to \omega_{*n}
#Zeff: Z effective due to impurties: Zeff=1/n_e(n_i+Z^2n_z)
# 


def VectorFinder_auto(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
    mu=abs(mu)
    if mu<=2:
        w0=VectorFinder_auto_tool(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
    else:
        w0=VectorFinder_auto_large_mu(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
    return w0

def VectorFinder_auto_Extensive(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
    mu=abs(mu)
    judge=0
    loopindex=0
    xmax=20.
    delx=0.02*abs(shat)/0.005
    
    guess_f=np.arange(0.5,3.+eta,0.3)
    guess_gamma=np.arange(0.1,1.,0.1)

    guess_mod=np.zeros(len(guess_f)*len(guess_gamma))
    guess_mod=np.array(guess_mod,dtype=complex)

    for i in range(len(guess_f)):
        for j in range(len(guess_gamma)):
            guess_mod[j*len(guess_gamma)+i]=guess_f[i] + 1j*guess_gamma[j]
    np.random.shuffle(guess_mod)
    
    w_list=[]
    odd_list=[]
    while judge == 0:
       
        print("Parameters:")
        print("(0) nu="+str(nu))
        print("(1) Zeff="+str(Zeff))
        print("(2) eta="+str(eta))
        print("(3) beta="+str(beta))
        print("(4) ky="+str(ky))
        print("(5) Modulation?="+str(ModIndex))
        print("(6) mu="+str(mu))
        print("(7) xstar="+str(xstar))
        print("(8) shat="+str(shat))
            
         
        xgrid=np.arange(-xmax,xmax,delx,complex)
        num=len(xgrid)
        b=np.ones(2*num-2)
        
        wguess=guess_mod[loopindex]

        
        w0=w_finder(xmax,delx,wguess,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar,1)
        A = A_maker(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        AInverse=np.linalg.inv(A)
        change=1.
        lold=2.
        while change > 10.**(-13):
            b=np.matmul(AInverse,b)
            b=b/np.linalg.norm(b)
            lnew=np.matmul(np.conj(b),np.matmul(A,b))
            change=np.abs(lnew-lold)
            lold=lnew
        Aparallel=b[0:num]
        ModG=np.abs(b[int(num/2)])*np.exp(-((xgrid-mu)/xstar)**2.)
        
        parity1,location0,ratio=parity_finder_short(xgrid,Aparallel,name='apar',plot=0,report=0)
        print('[oddness,eveness]='+str(ratio))

        #plt.clf()
        #plt.plot(xgrid,np.real(Aparallel))
        #plt.show()

        with open('./W_auto.log', 'a+') as csvfile:        #clear all and then write a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([guess_mod[loopindex],np.real(w0),np.imag(w0),\
                ratio[0],nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar])
        csvfile.close()

        #SigmaPlotter(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        loopindex=loopindex+1
        w_list.append(w0)
        odd_list.append(ratio[0])
        if ratio[0]<0.3 and np.imag(w0)>0:
            judge=1
        #elif loopindex>=2 and np.std(np.array(w_list))<0.02:
        #    print('std is'+str(np.std(np.array(w_list))))
        #    judge=1
        elif loopindex>=len(guess_mod):
            judge=1
        else:
            judge=0
    print('w_list='+str(w_list))
    print('odd_list='+str(odd_list))
    odd_w0s=[]
    odd_w0_real=[]
    for i in range(len(w_list)):
        if odd_list[i]<0.7:
            odd_w0s.append(w_list[i])
            odd_w0_real.append(w_list[i].imag)
    print(np.array(odd_w0_real))
    if len(odd_w0_real)==0:
        print('There is no mode in this')
        return 0.
    else:
        w0_index=np.argmax(np.array(odd_w0_real))
        w0=odd_w0s[w0_index]
        print('final w0='+str(w0))
        return w0

def VectorFinder_auto_tool(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
    judge=0
    loopindex=0
    xmax=20.
    delx=0.02 #for low magnetic shear
    guess_mod=[0.3,0.35,0.25,0.2,0.15,0.1]#modify the initial guess to( 1+eta+guess_mod[loopindex] )

    mu=abs(mu)
    
    w_list=[]
    odd_list=[]
    while judge == 0:
       
        print("Parameters:")
        print("(0) nu="+str(nu))
        print("(1) Zeff="+str(Zeff))
        print("(2) eta="+str(eta))
        print("(3) beta="+str(beta))
        print("(4) ky="+str(ky))
        print("(5) Modulation?="+str(ModIndex))
        print("(6) mu="+str(mu))
        print("(7) xstar="+str(xstar))
        print("(8) shat="+str(shat))
            
         
        xgrid=np.arange(-xmax,xmax,delx,complex)
        num=len(xgrid)
        b=np.ones(2*num-2)
        
        w0=1.+eta
        #wguess=w0+ Zeff*nu*0.1 + 1j*guess_mod[loopindex]
        wguess=w0 + 1j*guess_mod[loopindex]

        
        w0=w_finder(xmax,delx,wguess,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar,1)
        A = A_maker(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        AInverse=np.linalg.inv(A)
        change=1.
        lold=2.
        while change > 10.**(-13):
            b=np.matmul(AInverse,b)
            b=b/np.linalg.norm(b)
            lnew=np.matmul(np.conj(b),np.matmul(A,b))
            change=np.abs(lnew-lold)
            lold=lnew
        Aparallel=b[0:num]
        ModG=np.abs(b[int(num/2)])*np.exp(-((xgrid-mu)/xstar)**2.)
        
        parity1,location0,ratio=parity_finder_short(xgrid,Aparallel,name='apar',plot=0,report=0)
        print('[oddness,eveness]='+str(ratio))

        #plt.clf()
        #plt.plot(xgrid,np.real(Aparallel))
        #plt.show()

        with open('W_auto.log', 'a') as csvfile:        #clear all and then write a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([guess_mod[loopindex],w0,ratio[0],nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar])
        csvfile.close()

        #SigmaPlotter(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        loopindex=loopindex+1
        w_list.append(w0)
        odd_list.append(ratio[0])
        if ratio[0]<0.3 and np.imag(w0)>0:
            judge=1
        #elif loopindex>=2 and np.std(np.array(w_list))<0.02:
        #    print('std is'+str(np.std(np.array(w_list))))
        #    judge=1
        elif loopindex>=len(guess_mod):
            judge=1
        else:
            judge=0
    print('w_list='+str(w_list))
    print('odd_list='+str(odd_list))
    odd_w0s=[]
    odd_w0_real=[]
    for i in range(len(w_list)):
        if odd_list[i]<0.7:
            odd_w0s.append(w_list[i])
            odd_w0_real.append(w_list[i].imag)
    print(np.array(odd_w0_real))
    if len(odd_w0_real)==0:
        print('There is no mode in this')
        return 0.
    else:
        w0_index=np.argmax(np.array(odd_w0_real))
        w0=odd_w0s[w0_index]
        print('final w0='+str(w0))
        return w0
    
def VectorFinder_auto_tool_w0previous(w0_previous,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):

    judge=0
    loopindex=0
    xmax=20.
    delx=0.02
    guess_mod=[0.3,0.35,0.25,0.2,0.15,0.1]#modify the initial guess to( 1+eta+guess_mod[loopindex] )

    mu=abs(mu)
    
    w_list=[]
    odd_list=[]
    while judge == 0:
       
        print("Parameters:")
        print("(0) nu="+str(nu))
        print("(1) Zeff="+str(Zeff))
        print("(2) eta="+str(eta))
        print("(3) beta="+str(beta))
        print("(4) ky="+str(ky))
        print("(5) Modulation?="+str(ModIndex))
        print("(6) mu="+str(mu))
        print("(7) xstar="+str(xstar))
        print("(8) shat="+str(shat))
            
         
        xgrid=np.arange(-xmax,xmax,delx,complex)
        num=len(xgrid)
        b=np.ones(2*num-2)
        
        w0=1.+eta
        wguess=w0_previous+ Zeff*nu*0.1 + 1j*guess_mod[loopindex]

        
        w0=w_finder(xmax,delx,wguess,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar,1)
        A = A_maker(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        AInverse=np.linalg.inv(A)
        change=1.
        lold=2.
        while change > 10.**(-13):
            b=np.matmul(AInverse,b)
            b=b/np.linalg.norm(b)
            lnew=np.matmul(np.conj(b),np.matmul(A,b))
            change=np.abs(lnew-lold)
            lold=lnew
        Aparallel=b[0:num]
        ModG=np.abs(b[int(num/2)])*np.exp(-((xgrid-mu)/xstar)**2.)
        
        parity1,location0,ratio=parity_finder_short(xgrid,Aparallel,name='apar',plot=0,report=0)
        print('[oddness,eveness]='+str(ratio))

        #plt.clf()
        #plt.plot(xgrid,np.real(Aparallel))
        #plt.show()

        with open('W_auto.log', 'a') as csvfile:        #clear all and then write a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([guess_mod[loopindex],w0,ratio[0],nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar])
        csvfile.close()

        #SigmaPlotter(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        loopindex=loopindex+1
        w_list.append(w0)
        odd_list.append(ratio[0])
        if ratio[0]<0.3 and np.imag(w0)>0:
            judge=1
        #elif loopindex>=2 and np.std(np.array(w_list))<0.02:
        #    print('std is'+str(np.std(np.array(w_list))))
        #    judge=1
        elif loopindex>=len(guess_mod):
            judge=1
        else:
            judge=0
    print('w_list='+str(w_list))
    print('odd_list='+str(odd_list))
    odd_w0s=[]
    odd_w0_real=[]
    for i in range(len(w_list)):
        if odd_list[i]<0.7:
            odd_w0s.append(w_list[i])
            odd_w0_real.append(w_list[i].imag)
    print(np.array(odd_w0_real))
    if len(odd_w0_real)==0:
        print('There is no mode in this')
        return 0.
    else:
        w0_index=np.argmax(np.array(odd_w0_real))
        w0=odd_w0s[w0_index]
        print('final w0='+str(w0))
        return w0

def VectorFinder_auto_large_mu(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
    mu=abs(mu)
    mu0=0.
    delta_mu=0.5
    loop_index=0
    w0_temp=0
    while(1==1):
        if loop_index==1:
            w0_temp=VectorFinder_auto_tool(nu,Zeff,eta,shat,beta,ky,ModIndex,mu0,xstar)
            if abs(w0_temp)<0.001:
                w0_temp=1.+eta
            w0=w0_temp
        else:
            w0=VectorFinder_auto_tool_w0previous(w0_temp,nu,Zeff,eta,shat,beta,ky,ModIndex,mu0,xstar)
            if abs(w0)<0.001:
                w0=1.+eta
            w0_temp=w0
        gamma=w0.imag
        if mu0<mu:
            mu0=mu0+delta_mu
            if mu0>mu:
                mu0=mu
        else:#reach to the final mu
            break

        if gamma<0:#reach to the growth that is stable
            w0=-1j*100
            break

        loop_index=loop_index+1
    return w0

def Gaussian(sigma,mu,x_list):
	return 1./(sigma * np.sqrt(2. * np.pi)) * np.exp( - (x_list - mu)**2. / (2. * sigma**2.) )

def A_maker(x_max, del_x, w1, v1,Zeff,eta,alpha,beta,ky,ModIndex,mu,xstar):
    
    mref=2.
    BC=(1.0-0.5*ky*del_x)/(1.+0.5*ky*del_x)
    #BC=0
    BC=0.
    # making grid
    x_min = -x_max
    x_grid = np.arange(x_min, x_max+del_x, del_x)
    num = len(x_grid)
    # initializing matrix A
    A = np.zeros((2*num-4, 2*num-4), dtype=complex)
    
    # Calling the conductivity function which defines the conductvity order that will be used. 
    # The code also converts conversion of the data types
    
    #L_maker(leg, lag) ######### calling a function
    w_hat = w1/v1
    
    ### (a bunch of code that would've gone here was commented out in the Mathematica notebook by Joel
    ### so I'm skipping it for now)
    SMinusArray=np.array([[[0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.,0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.]], [[0.707107, -0.57735, 0., 0., 0., 0., 0., 0.], [0., 0.912871, -0.816497, 0., 0., 0., 0., 0.], [0., 0., 1.08012, -1., 0., 0., 0., 0.], [0., 0., 0., 1.22474, -1.1547, 0., 0., 0.], [0., 0., 0., 0., 1.35401, -1.29099, 0., 0.], [0., 0., 0., 0., 0., 1.47196, -1.41421, 0.], [0., 0., 0., 0., 0., 0., 1.58114, -1.52753], [0., 0., 0., 0., 0., 0., 0., 1.68325]], [[0.816497, -0.516398, 0., 0., 0., 0., 0., 0.], [0., 0.966092, -0.730297, 0., 0., 0., 0., 0.], [0., 0., 1.09545, -0.894427, 0., 0., 0., 0.], [0., 0., 0., 1.21106, -1.0328, 0., 0., 0.], [0., 0., 0., 0., 1.31656, -1.1547, 0., 0.], [0., 0., 0., 0., 0., 1.41421, -1.26491, 0.], [0., 0., 0., 0., 0., 0., 1.50555, -1.36626], [0., 0., 0., 0., 0., 0., 0., 1.59164]], [[0.948683, -0.507093, 0., 0., 0., 0., 0., 0.], [0., 1.07571, -0.717137, 0., 0., 0., 0., 0.], [0., 0., 1.18924, -0.87831, 0., 0., 0., 0.], [0., 0., 0., 1.29284, -1.01419, 0., 0., 0.], [0., 0., 0., 0., 1.38873, -1.13389, 0., 0.], [0., 0., 0., 0., 0., 1.47842, -1.24212, 0.], [0., 0., 0., 0., 0., 0., 1.56296, -1.34164], [0., 0., 0., 0., 0., 0., 0., 1.64317]], [[1.06904, -0.503953, 0., 0., 0., 0., 0., 0.], [0., 1.18187, -0.712697, 0., 0., 0., 0., 0.], [0., 0., 1.28483, -0.872872, 0., 0., 0., 0.], [0., 0., 0., 1.38013, -1.00791, 0., 0., 0.], [0., 0., 0., 0., 1.46926, -1.12687, 0., 0.], [0., 0., 0., 0., 0., 1.55329, -1.23443, 0.], [0., 0., 0., 0., 0., 0., 1.63299, -1.33333], [0., 0., 0., 0., 0., 0., 0., 1.70899]], [[1.17851, -0.502519, 0., 0., 0., 0., 0., 0.], [0., 1.28118, -0.710669, 0., 0., 0., 0., 0.], [0., 0., 1.3762, -0.870388, 0., 0., 0., 0.], [0., 0., 0., 1.46508, -1.00504, 0., 0., 0.], [0., 0., 0., 0., 1.54887, -1.12367, 0., 0.], [0., 0., 0., 0., 0., 1.62835, -1.23091, 0.], [0., 0., 0., 0., 0., 0., 1.70412, -1.32954], [0., 0., 0., 0., 0., 0., 0., 1.77667]]])
    SMinusArray=SMinusArray.astype(complex)
    SPlusArray=np.array([[[0.707107, 0., 0., 0., 0., 0., 0., 0.], [-0.57735, 0.912871, 0., 0., 0., 0., 0., 0.], [0., -0.816497, 1.08012, 0., 0., 0., 0., 0.], [0., 0., -1., 1.22474, 0., 0., 0., 0.], [0., 0., 0., -1.1547, 1.35401, 0., 0., 0.], [0., 0., 0., 0., -1.29099, 1.47196, 0., 0.], [0., 0., 0., 0., 0., -1.41421, 1.58114, 0.], [0., 0., 0., 0., 0., 0., -1.52753, 1.68325]], [[0.816497, 0., 0., 0., 0., 0., 0., 0.], [-0.516398, 0.966092, 0., 0., 0., 0., 0., 0.], [0., -0.730297, 1.09545, 0., 0., 0., 0., 0.], [0., 0., -0.894427, 1.21106, 0., 0., 0., 0.], [0., 0., 0., -1.0328, 1.31656, 0., 0., 0.], [0., 0., 0., 0., -1.1547, 1.41421, 0., 0.], [0., 0., 0., 0., 0., -1.26491, 1.50555, 0.], [0., 0., 0., 0., 0., 0., -1.36626, 1.59164]], [[0.948683, 0., 0., 0., 0., 0., 0., 0.], [-0.507093, 1.07571, 0., 0., 0., 0., 0., 0.], [0., -0.717137, 1.18924, 0., 0., 0., 0., 0.], [0., 0., -0.87831, 1.29284, 0., 0., 0., 0.], [0., 0., 0., -1.01419, 1.38873, 0., 0., 0.], [0., 0., 0., 0., -1.13389, 1.47842, 0., 0.], [0., 0., 0., 0., 0., -1.24212, 1.56296, 0.], [0., 0., 0., 0., 0., 0., -1.34164, 1.64317]], [[1.06904, 0., 0., 0., 0., 0., 0., 0.], [-0.503953, 1.18187, 0., 0., 0., 0., 0., 0.], [0., -0.712697, 1.28483, 0., 0., 0., 0., 0.], [0., 0., -0.872872, 1.38013, 0., 0., 0., 0.], [0., 0., 0., -1.00791, 1.46926, 0., 0., 0.], [0., 0., 0., 0., -1.12687, 1.55329, 0., 0.], [0., 0., 0., 0., 0., -1.23443, 1.63299, 0.], [0., 0., 0., 0., 0., 0., -1.33333, 1.70899]], [[1.17851, 0., 0., 0., 0., 0., 0., 0.], [-0.502519, 1.28118, 0., 0., 0., 0., 0., 0.], [0., -0.710669, 1.3762, 0., 0., 0., 0., 0.], [0., 0., -0.870388, 1.46508, 0., 0., 0., 0.], [0., 0., 0., -1.00504, 1.54887, 0., 0., 0.], [0., 0., 0., 0., -1.12367, 1.62835, 0., 0.], [0., 0., 0., 0., 0., -1.23091, 1.70412, 0.], [0., 0., 0., 0., 0., 0., -1.32954, 1.77667]]])
    SPlusArray=SPlusArray.astype(complex)
    VArray=np.array([[[-0.752253, -0.71365, -0.674336, -0.642358, -0.61628, -0.594565, \
        -0.576111, -0.560156], [-0.71365, -1.40347, -1.22218, -1.08074, \
        -0.992364, -0.93612, -0.897748, -0.869373], [-0.674336, -1.22218, \
        -1.84721, -1.66056, -1.46245, -1.31971, -1.22419, -1.16037], \
        [-0.642358, -1.08074, -1.66056, -2.21962, -2.03559, -1.80662, \
        -1.62394, -1.49317], [-0.61628, -0.992364, -1.46245, -2.03559, \
        -2.54542, -2.36554, -2.11866, -1.907], [-0.594565, -0.93612, \
        -1.31971, -1.80662, -2.36554, -2.83783, -2.66232, -2.40462], \
        [-0.576111, -0.897748, -1.22419, -1.62394, -2.11866, -2.66232, \
        -3.10496, -2.93366], [-0.560156, -0.869373, -1.16037, -1.49317, \
        -1.907, -2.40462, -2.93366, -3.35218]], [[-1.54101, -0.979665, \
        -0.709766, -0.565628, -0.47891, -0.421029, -0.379071, -0.34673], \
        [-0.979665, -1.87503, -1.4639, -1.14456, -0.936081, -0.799501, \
        -0.70615, -0.638799], [-0.709766, -1.4639, -2.20379, -1.86397, \
        -1.52646, -1.27529, -1.09786, -0.972146], [-0.565628, -1.14456, \
        -1.86397, -2.50873, -2.21109, -1.86757, -1.58805, -1.37842], \
        [-0.47891, -0.936081, -1.52646, -2.21109, -2.79014, -2.52079, \
        -2.17666, -1.87794], [-0.421029, -0.799501, -1.27529, -1.86757, \
        -2.52079, -3.05116, -2.80237, -2.4602], [-0.379071, -0.70615, \
        -1.09786, -1.58805, -2.17666, -2.80237, -3.29493, -3.06191], \
        [-0.34673, -0.638799, -0.972146, -1.37842, -1.87794, -2.4602, \
        -3.06191, -3.52407]], [[-1.98912, -1.16473, -0.750363, -0.529727, \
        -0.402577, -0.323368, -0.27032, -0.232475], [-1.16473, -2.25162, \
        -1.64859, -1.19363, -0.898424, -0.707261, -0.579768, -0.491176], \
        [-0.750363, -1.64859, -2.52093, -2.03276, -1.57436, -1.23727, \
        -1.00006, -0.832966], [-0.529727, -1.19363, -2.03276, -2.78138, \
        -2.36427, -1.91294, -1.55095, -1.27991], [-0.402577, -0.898424, \
        -1.57436, -2.36427, -3.02927, -2.66072, -2.21975, -1.84264], \
        [-0.323368, -0.707261, -1.23727, -1.91294, -2.66072, -3.26446, \
        -2.93134, -2.50148], [-0.27032, -0.579768, -1.00006, -1.55095, \
        -2.21975, -2.93134, -3.48782, -3.18179], [-0.232475, -0.491176, \
        -0.832966, -1.27991, -1.84264, -2.50148, -3.18179, -3.70047]], \
        [[-2.30639, -1.27914, -0.771783, -0.504612, -0.35385, -0.263206, \
        -0.205343, -0.166271], [-1.27914, -2.5513, -1.77966, -1.22098, \
        -0.865687, -0.639701, -0.492387, -0.393094], [-0.771783, -1.77966, \
        -2.7927, -2.16365, -1.60287, -1.20017, -0.921429, -0.728256], \
        [-0.504612, -1.22098, -2.16365, -3.02675, -2.49012, -1.94127, \
        -1.51175, -1.19505], [-0.35385, -0.865687, -1.60287, -2.49012, \
        -3.2519, -2.78036, -2.24762, -1.80292], [-0.263206, -0.639701, \
        -1.20017, -1.94127, -2.78036, -3.46791, -3.04478, -2.52893], \
        [-0.205343, -0.492387, -0.921429, -1.51175, -2.24762, -3.04478, \
        -3.67511, -3.28946], [-0.166271, -0.393094, -0.728256, -1.19505, \
        -1.80292, -2.52893, -3.28946, -3.87407]], [[-2.55498, -1.34775, \
        -0.774978, -0.480505, -0.317789, -0.2224, -0.163458, -0.125198], \
        [-1.34775, -2.79575, -1.86896, -1.22832, -0.831993, -0.585316, \
        -0.427982, -0.324588], [-0.774978, -1.86896, -3.02436, -2.26068, \
        -1.61281, -1.16071, -0.854406, -0.6462], [-0.480505, -1.22832, \
        -2.26068, -3.24353, -2.58906, -1.95292, -1.46903, -1.11951], \
        [-0.317789, -0.831993, -1.61281, -2.58906, -3.45414, -2.87847, \
        -2.26049, -1.75865], [-0.2224, -0.585316, -1.16071, -1.95292, \
        -2.87847, -3.65678, -3.14081, -2.54277], [-0.163458, -0.427982, \
        -0.854406, -1.46903, -2.26049, -3.14081, -3.85198, -3.38286], \
        [-0.125198, -0.324588, -0.6462, -1.11951, -1.75865, -2.54277, \
        -3.38286, -4.04026]]])
            
    VArray=VArray.astype(complex)
    VZArray=(Zeff-1)*np.array([[[-0.752253, -0.71365, -0.674336, -0.642358, -0.61628, -0.594565, \
        -0.576111, -0.560156], [-0.71365, -0.977929, -0.980921, -0.95762, \
        -0.931117, -0.905905, -0.882884, -0.862067], [-0.674336, -0.980921, \
        -1.16331, -1.18126, -1.16978, -1.15031, -1.12894, -1.10779], \
        [-0.642358, -0.95762, -1.18126, -1.32405, -1.34877, -1.34541, \
        -1.33201, -1.31482], [-0.61628, -0.931117, -1.16978, -1.34877, \
        -1.46783, -1.49592, -1.49821, -1.48976], [-0.594565, -0.905905, \
        -1.15031, -1.34541, -1.49592, -1.59904, -1.62884, -1.63519], \
        [-0.576111, -0.882884, -1.12894, -1.33201, -1.49821, -1.62884, \
        -1.72044, -1.75109], [-0.560156, -0.862067, -1.10779, -1.31482, \
        -1.48976, -1.63519, -1.75109, -1.83395]], [[-0.902703, -0.723773, \
        -0.603144, -0.519693, -0.458641, -0.411927, -0.374928, -0.344825], \
        [-0.723773, -1.09614, -0.999422, -0.892889, -0.80356, -0.730611, \
        -0.670595, -0.620533], [-0.603144, -0.999422, -1.26271, -1.20441, \
        -1.11584, -1.03142, -0.956841, -0.892046], [-0.519693, -0.892889, \
        -1.20441, -1.41096, -1.37453, -1.30132, -1.22444, -1.1523], \
        [-0.458641, -0.80356, -1.11584, -1.37453, -1.54575, -1.52316, \
        -1.46236, -1.39326], [-0.411927, -0.730611, -1.03142, -1.30132, \
        -1.52316, -1.67011, -1.65689, -1.60609], [-0.374928, -0.670595, \
        -0.956841, -1.22444, -1.46236, -1.65689, -1.78612, -1.77954], \
        [-0.344825, -0.620533, -0.892046, -1.1523, -1.39326, -1.60609, \
        -1.77954, -1.89523]], [[-1.03166, -0.729494, -0.549877, -0.435829, \
        -0.35807, -0.30209, -0.260083, -0.227525], [-0.729494, -1.2036, \
        -1.01094, -0.836483, -0.703316, -0.601992, -0.523427, -0.461203], \
        [-0.549877, -1.01094, -1.35601, -1.2199, -1.06332, -0.92868, \
        -0.817984, -0.727256], [-0.435829, -0.836483, -1.2199, -1.49413, \
        -1.3927, -1.25433, -1.12467, -1.01166], [-0.35807, -0.703316, \
        -1.06332, -1.3927, -1.62127, -1.54314, -1.42097, -1.29864], \
        [-0.30209, -0.601992, -0.92868, -1.25433, -1.54314, -1.73962, \
        -1.67811, -1.56987], [-0.260083, -0.523427, -0.817984, -1.12467, \
        -1.42097, -1.67811, -1.85076, -1.80161], [-0.227525, -0.461203, \
        -0.727256, -1.01166, -1.29864, -1.56987, -1.80161, -1.95583]], \
        [[-1.14629, -0.73317, -0.508362, -0.375102, -0.289483, -0.231014, \
        -0.189183, -0.158141], [-0.73317, -1.3026, -1.0188, -0.788296, \
        -0.624037, -0.505956, -0.418852, -0.352892], [-0.508362, -1.0188, \
        -1.44388, -1.231, -1.01523, -0.842031, -0.707221, -0.60185], \
        [-0.375102, -0.788296, -1.231, -1.57364, -1.40623, -1.20882, \
        -1.03625, -0.893625], [-0.289483, -0.624037, -1.01523, -1.40623, \
        -1.69423, -1.55846, -1.37891, -1.21135], [-0.231014, -0.505956, \
        -0.842031, -1.20882, -1.55846, -1.80729, -1.69478, -1.53145], \
        [-0.189183, -0.418852, -0.707221, -1.03625, -1.37891, -1.69478, \
        -1.91405, -1.81929], [-0.158141, -0.352892, -0.60185, -0.893625, \
        -1.21135, -1.53145, -1.81929, -2.01544]], [[-1.2505, -0.735729, \
        -0.474911, -0.329162, -0.240287, -0.182395, -0.142726, -0.114435], \
        [-0.735729, -1.39479, -1.02451, -0.746982, -0.560254, -0.4325, \
        -0.342349, -0.276792], [-0.474911, -1.02451, -1.52705, -1.23937, \
        -0.971945, -0.768963, -0.618229, -0.505249], [-0.329162, -0.746982, \
        -1.23937, -1.64977, -1.4167, -1.16619, -0.958841, -0.795015], \
        [-0.240287, -0.560254, -0.971945, -1.4167, -1.76468, -1.57061, \
        -1.33814, -1.13256], [-0.182395, -0.4325, -0.768963, -1.16619, \
        -1.57061, -1.87306, -1.70825, -1.49306], [-0.142726, -0.342349, \
        -0.618229, -0.958841, -1.33814, -1.70825, -1.97588, -1.83379], \
        [-0.114435, -0.276792, -0.505249, -0.795015, -1.13256, -1.49306, \
        -1.83379, -2.0739]]])
    VZArray=VZArray.astype(complex)
    VArray=VArray+VZArray
    L11_grid=np.arange(x_min, x_max+del_x, del_x,dtype=complex)
    L12grid=np.arange(x_min, x_max+del_x, del_x,dtype=complex)
    for i in range(num):
        k= (np.sqrt(2)*x_grid[i]*alpha*np.sqrt(mref*1836.))/v1
        h1 = np.linalg.inv(1j*w_hat*np.identity(8)+VArray[0]-1j*(k**2/w_hat)*np.matmul(SMinusArray[1],SPlusArray[0])+k**2*np.matmul(np.matmul(SPlusArray[1],np.linalg.inv(
                1j*w_hat*np.identity(8)+VArray[1]+ 
                k**2*np.matmul(np.matmul(SPlusArray[2],np.linalg.inv(
                        1j*w_hat*np.identity(8)+VArray[2]+
                        k**2*np.matmul(np.matmul(SPlusArray[3],np.linalg.inv(
                                1j*w_hat*np.identity(8)+VArray[3]+
                                k**2*np.matmul(np.matmul(SPlusArray[4],np.linalg.inv(
                                        1j*w_hat*np.identity(8)+VArray[4])),SMinusArray[5]))),SMinusArray[
                                        4]))),SMinusArray[3]))),SMinusArray[2]))
        L11_grid[i]=np.matmul(h1,np.array([0.5+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0,0,0],dtype=complex))[0]
        L12grid[i]=np.matmul(h1,np.array([0.+0.j,-0.790569+0.j,0,0,0,0,0,0],dtype=complex))[0]
    
    if ModIndex==0:
        ModG=1.
    
    elif ModIndex==1:
        ModG=np.exp(-((x_grid-mu)/xstar)**2)
    else:
        print("ModIndex must be 0 or 1")
        ModG=0
    sigma_grid = (w1*L11_grid-(1.0+eta)*np.multiply(L11_grid,ModG) - eta*np.multiply(L12grid,ModG))/v1
    #print(sigma_grid)
    #print(ModG)
    # computing the diagonal components of the matrix
    #    a11 = ky**2 + 2j*mref*1836.*beta*sigma_grid
    #    a12 = -4j*(mref*1836)**1.5*alpha*beta*sigma_grid*x_grid
    #    a21 = 4j*alpha*np.sqrt(mref*1836.)*sigma_grid/(w1*(w1-1))*x_grid
    #    a22 = ky**2 - 8j*alpha**2*mref*1836.*sigma_grid/(w1*(w1-1))*x_grid**2
    tau=+1.0
    a11=ky**2 +1j*mref*1836.*beta*sigma_grid
    a12=-1j*mref*1836*beta*sigma_grid*x_grid
    a21=2j*mref*1836*(alpha**2)*sigma_grid/(w1*(w1+tau*ModG))*x_grid
    a22=ky**2-2j*mref*1836*(alpha**2)*sigma_grid/(w1*(w1+tau*ModG))*x_grid**2
    # populating the matrix with the components of the matrix
    # this loop populates the off-diagonal components coming from the finite difference
    for i in range(num-3):
        A[i, i+1], A[i+1, i], A[num-2+i, num-2+i+1],  A[num-2+i+1, num-2+i] \
        = -1/del_x**2, -1/del_x**2, -1/del_x**2, -1/del_x**2

      # this loop populates the diagonal components of the matrix
      ##### testing
    for i in range(num-2):
        A[i,i] = 2/del_x**2 + a11[i+1]
        A[num-2+i, num-2+i] = 2/del_x**2 + a22[i+1]
        A[num-2+i, i] = a21[i+1]
        A[i, num-2+i] = a12[i+1]

    A[0,0] = A[0,0] - 1/del_x**2*BC
    A[num-3,num-3] = A[num-3,num-3] - 1/del_x**2*BC
    A[num-2,num-2] = A[num-2,num-2] - 1/del_x**2*BC
    A[2*num-5,2*num-5] =  A[2*num-5,2*num-5] - 1/del_x**2*BC
    return A

def w_finder(x_max, del_x, w_guess, v,Zeff,ne,alpha,beta,ky,ModIndex,mu,xstar,printIndex):

    w_minus=w_guess
    # call A_maker to create and populate matrix A
    A = A_maker(x_max, del_x, w_guess, v,Zeff,ne,alpha,beta,ky,ModIndex,mu,xstar) ##### maybe relabel this as A-minus

    # first step is chosen to be del_w = 0.01 (??? should this be a parameter?)
    del_w = 0.01j
    det_A_minus = np.linalg.slogdet(A)
    w0 = w_minus + del_w

    # iterative loop that runs until the correction to the root is very small
    # secant method??
    neg_streak=0
    while np.abs(del_w) > 10**-8:
        A = A_maker(x_max, del_x, w0, v,Zeff,ne,alpha,beta,ky,ModIndex,mu,xstar)
        det_A0 = np.linalg.slogdet(A)
        del_w = -del_w/(1-(det_A_minus[0]/det_A0[0])*np.exp(det_A_minus[1]-det_A0[1]))
        w_plus = w0 + del_w
        w_minus = w0
        w0 = w_plus
        det_A_minus = det_A0
        if w0.imag<0:
            neg_streak=neg_streak+1
        else:
            neg_streak=0

        if neg_streak>=4:
            break

        if printIndex==1:
            print(w0)
    return w0

def VectorFinder(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
#    mref=2.
#    xsigma=1/shat*np.sqrt(1./(mref*1836))
#    xmax=xsigma*35
#    delx=xsigma/50
#    xmax=20.
#    delx=0.02
    judge=0
    loopindex=0
    xmax=float(input("(x) xmax="))
    delx=float(input("(x) delx="))
    while judge == 0:
        
       
        print("Parameters:")
        print("(0) nu="+str(nu))
        print("(1) Zeff="+str(Zeff))
        print("(2) eta="+str(eta))
        print("(3) beta="+str(beta))
        print("(4) ky="+str(ky))
        print("(5) Modulation?="+str(ModIndex))
        print("(6) mu="+str(mu))
        print("(7) xstar="+str(xstar))
        print("(8) shat="+str(shat))
        changeParameters=int(input("Would you like to change parameters? 0 signifies no. 1 signifies yes: "))
        while changeParameters!=0:
            which=int(input('Enter the parameters indices to change: '))
            if which==0:
                nu=float(input("nu="))
            elif which ==1:
                Zeff=float(input("Zeff="))
            elif which==2:
                eta=float(input("eta"))
            elif which==3:
                beta=float(input("beta="))
            elif which==4:
                ky=float(input("ky="))
            elif which==5:
                ModIndex=float(input("ModIndex="))
            elif which==6:
                mu=float(input("mu="))
            elif which==7:
                xstar=float(input("xstar="))
            elif which==8:
                shat=float(input("shat="))
            changeParameters=int(input('Are there other parameters to change? 0 signifies no. 1 signifies yes: '))
            
        changeGrid=int(input("Would you like to change the grid? 0 signifies no. 1 signifies yes: "))
        if changeGrid==1:
            print("Current Values:")
            print("(0) xmax="+str(xmax))
            print("(1) delx="+str(delx))
            print("Enter new values:")
            xmax=float(input("xmax="))
            delx=float(input("delx="))
        #wguess=3.1902795704258415+0.08383523830945384j
        wguess=complex(input("Enter initial guess for omega (enter 0 to use previous or 1 to use default): "))
        xgrid=np.arange(-xmax,xmax,delx,complex)
        num=len(xgrid)
        b=np.ones(2*num-2)
        if wguess==0:
            if loopindex==0:
                w0=1.+eta
            wguess=w0
        if wguess==1:
            wguess=1.+eta
        w0=w_finder(xmax,delx,wguess,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar,1)
        A = A_maker(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        AInverse=np.linalg.inv(A)
        change=1.
        lold=2.
        while change > 10.**(-13):
            b=np.matmul(AInverse,b)
            b=b/np.linalg.norm(b)
            lnew=np.matmul(np.conj(b),np.matmul(A,b))
            change=np.abs(lnew-lold)
            lold=lnew
        Aparallel=b[0:num]
        ModG=np.abs(b[int(num/2)])*np.exp(-((xgrid-mu)/xstar)**2)
        plt.plot(xgrid,np.real(Aparallel),label="Re(Aparallel)")
        plt.plot(xgrid,np.imag(Aparallel),label="Im(Aparallel)")
        plt.plot(xgrid,ModG,label="Omega*(x)")
        plt.legend()
        plt.show()
        SigmaPlotter(xmax, delx, w0, nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        loopindex=loopindex+1
        judge=int(input("Would you like to continue searching for modes?: To exit routine enter 1 and to repeat the routine enter 0: "))
    
    return w0

def ParameterScan():
    print("Initializing Parameters: ")
    print("Please input all initial parameters:")
    
    nu=float(input("(0) nu="))
    Zeff=float(input("(x) Zeff="))
    eta=float(input("(1) eta="))
    shat=float(input("(2) shat="))
    beta=float(input("(3) beta="))
    ky=float(input("(4) ky="))
    ModIndex=float(input("(x) ModIndex="))
    mu=float(input("(5) mu="))
    xstar=float(input("(x) xstar="))
    xmax=float(input("(x) xmax="))
    delx=float(input("(x) delx="))
    print("Now running initialization scheme... ")
    w0=VectorFinder(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
    paramindex=int(input("Which parameter would you like to scan? (enter parameter index): "))
    if paramindex==0:
        scannedParam="nu"
        print("Please enter final scan value and step size: ")
        finalParam=float(input("Final Value for nu: "))
        delParam=float(input("Step size: "))
        arrayParam=np.arange(nu,finalParam,delParam)
        w0Array=np.zeros(len(arrayParam),dtype=complex)
        for i in range(0,len(arrayParam)):
            w0=w_finder(xmax,delx,w0,arrayParam[i],Zeff,eta,shat,beta,ky,ModIndex,mu,xstar,0)
            w0Array[i]=w0
            print("nu: "+str(arrayParam[i]))
            print('w0: '+str(w0))
    if paramindex==1:
        scannedParam="eta"
        print("Please enter final scan value and step size: ")
        finalParam=float(input("Final Value for eta: "))
        delParam=float(input("Step size: "))
        arrayParam=np.arange(eta,finalParam,delParam)
        w0Array=np.zeros(len(arrayParam),dtype=complex)
        for i in range(0,len(arrayParam)):
            w0=w_finder(xmax,delx,w0,nu,Zeff,arrayParam,shat,beta,ky,ModIndex,mu,xstar,0)
            w0Array[i]=w0
            print("eta: "+str(arrayParam[i]))
            print('w0: '+str(w0))
    if paramindex==2:
        scannedParam="shat"
        print("Please enter final scan value and step size: ")
        finalParam=float(input("Final Value for shat: "))
        delParam=float(input("Step size: "))
        arrayParam=np.arange(shat,finalParam,delParam)
        w0Array=np.zeros(len(arrayParam),dtype=complex)
        for i in range(0,len(arrayParam)):
            w0=w_finder(xmax,delx,w0,nu,Zeff,eta,arrayParam[i],beta,ky,ModIndex,mu,xstar,0)
            w0Array[i]=w0
            print("shat: "+str(arrayParam[i]))
            print('w0: '+str(w0))
    if paramindex==3:
        scannedParam="beta"
        print("Please enter final scan value and step size: ")
        finalParam=float(input("Final Value for beta: "))
        delParam=float(input("Step size: "))
        arrayParam=np.arange(beta,finalParam,delParam)
        w0Array=np.zeros(len(arrayParam),dtype=complex)
        for i in range(0,len(arrayParam)):
            w0=w_finder(xmax,delx,w0,nu,Zeff,eta,shat,arrayParam[i],ky,ModIndex,mu,xstar,0)
            w0Array[i]=w0
            print("beta: "+str(arrayParam[i]))
            print('w0: '+str(w0))
    if paramindex==4:
        scannedParam="ky"
        print("Please enter final scan value and step size: ")
        finalParam=float(input("Final Value for ky: "))
        delParam=float(input("Step size: "))
        arrayParam=np.arange(ky,finalParam,delParam)
        w0Array=np.zeros(len(arrayParam),dtype=complex)
        for i in range(0,len(arrayParam)):
            w0=w_finder(xmax,delx,w0,nu,Zeff,eta,shat,beta,arrayParam[i],ModIndex,mu,xstar,0)
            w0Array[i]=w0
            print("ky: "+str(arrayParam[i]))
            print('w0: '+str(w0))
    if paramindex==5:
        scannedParam="mu"
        print("Please enter final scan value and step size: ")
        finalParam=float(input("Final Value for mu: "))
        delParam=float(input("Step size: "))
        arrayParam=np.arange(mu,finalParam,delParam)
        w0Array=np.zeros(len(arrayParam),dtype=complex)
        for i in range(0,len(arrayParam)):
            w0=w_finder(xmax,delx,w0,nu,Zeff,eta,shat,beta,ky,ModIndex,arrayParam[i],xstar,0)
            w0Array[i]=w0
            print("mu: "+str(arrayParam[i]))
            print('w0: '+str(w0))
    
    plt.plot(arrayParam,np.imag(w0Array),label="gamma")
    plt.show()
    plt.plot(arrayParam,np.real(w0Array),label="Re(omega)")
    plt.show()
    
    
    
def SigmaPlotter(x_max, del_x, w1, v1,Zeff,eta,alpha,beta,ky,ModIndex,mu,xstar):
    x_min = -x_max
    x_grid = np.arange(x_min, x_max+del_x, del_x)
    num = len(x_grid)
    mref=2.
    # initializing matrix A
    
    # Calling the conductivity function which defines the conductvity order that will be used. 
    # The code also converts conversion of the data types
    
    #L_maker(leg, lag) ######### calling a function
    w_hat = w1/v1
    
    ### (a bunch of code that would've gone here was commented out in the Mathematica notebook by Joel
    ### so I'm skipping it for now)
    SMinusArray=np.array([[[0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.,0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.]], [[0.707107, -0.57735, 0., 0., 0., 0., 0., 0.], [0., 0.912871, -0.816497, 0., 0., 0., 0., 0.], [0., 0., 1.08012, -1., 0., 0., 0., 0.], [0., 0., 0., 1.22474, -1.1547, 0., 0., 0.], [0., 0., 0., 0., 1.35401, -1.29099, 0., 0.], [0., 0., 0., 0., 0., 1.47196, -1.41421, 0.], [0., 0., 0., 0., 0., 0., 1.58114, -1.52753], [0., 0., 0., 0., 0., 0., 0., 1.68325]], [[0.816497, -0.516398, 0., 0., 0., 0., 0., 0.], [0., 0.966092, -0.730297, 0., 0., 0., 0., 0.], [0., 0., 1.09545, -0.894427, 0., 0., 0., 0.], [0., 0., 0., 1.21106, -1.0328, 0., 0., 0.], [0., 0., 0., 0., 1.31656, -1.1547, 0., 0.], [0., 0., 0., 0., 0., 1.41421, -1.26491, 0.], [0., 0., 0., 0., 0., 0., 1.50555, -1.36626], [0., 0., 0., 0., 0., 0., 0., 1.59164]], [[0.948683, -0.507093, 0., 0., 0., 0., 0., 0.], [0., 1.07571, -0.717137, 0., 0., 0., 0., 0.], [0., 0., 1.18924, -0.87831, 0., 0., 0., 0.], [0., 0., 0., 1.29284, -1.01419, 0., 0., 0.], [0., 0., 0., 0., 1.38873, -1.13389, 0., 0.], [0., 0., 0., 0., 0., 1.47842, -1.24212, 0.], [0., 0., 0., 0., 0., 0., 1.56296, -1.34164], [0., 0., 0., 0., 0., 0., 0., 1.64317]], [[1.06904, -0.503953, 0., 0., 0., 0., 0., 0.], [0., 1.18187, -0.712697, 0., 0., 0., 0., 0.], [0., 0., 1.28483, -0.872872, 0., 0., 0., 0.], [0., 0., 0., 1.38013, -1.00791, 0., 0., 0.], [0., 0., 0., 0., 1.46926, -1.12687, 0., 0.], [0., 0., 0., 0., 0., 1.55329, -1.23443, 0.], [0., 0., 0., 0., 0., 0., 1.63299, -1.33333], [0., 0., 0., 0., 0., 0., 0., 1.70899]], [[1.17851, -0.502519, 0., 0., 0., 0., 0., 0.], [0., 1.28118, -0.710669, 0., 0., 0., 0., 0.], [0., 0., 1.3762, -0.870388, 0., 0., 0., 0.], [0., 0., 0., 1.46508, -1.00504, 0., 0., 0.], [0., 0., 0., 0., 1.54887, -1.12367, 0., 0.], [0., 0., 0., 0., 0., 1.62835, -1.23091, 0.], [0., 0., 0., 0., 0., 0., 1.70412, -1.32954], [0., 0., 0., 0., 0., 0., 0., 1.77667]]])
    SMinusArray=SMinusArray.astype(complex)
    SPlusArray=np.array([[[0.707107, 0., 0., 0., 0., 0., 0., 0.], [-0.57735, 0.912871, 0., 0., 0., 0., 0., 0.], [0., -0.816497, 1.08012, 0., 0., 0., 0., 0.], [0., 0., -1., 1.22474, 0., 0., 0., 0.], [0., 0., 0., -1.1547, 1.35401, 0., 0., 0.], [0., 0., 0., 0., -1.29099, 1.47196, 0., 0.], [0., 0., 0., 0., 0., -1.41421, 1.58114, 0.], [0., 0., 0., 0., 0., 0., -1.52753, 1.68325]], [[0.816497, 0., 0., 0., 0., 0., 0., 0.], [-0.516398, 0.966092, 0., 0., 0., 0., 0., 0.], [0., -0.730297, 1.09545, 0., 0., 0., 0., 0.], [0., 0., -0.894427, 1.21106, 0., 0., 0., 0.], [0., 0., 0., -1.0328, 1.31656, 0., 0., 0.], [0., 0., 0., 0., -1.1547, 1.41421, 0., 0.], [0., 0., 0., 0., 0., -1.26491, 1.50555, 0.], [0., 0., 0., 0., 0., 0., -1.36626, 1.59164]], [[0.948683, 0., 0., 0., 0., 0., 0., 0.], [-0.507093, 1.07571, 0., 0., 0., 0., 0., 0.], [0., -0.717137, 1.18924, 0., 0., 0., 0., 0.], [0., 0., -0.87831, 1.29284, 0., 0., 0., 0.], [0., 0., 0., -1.01419, 1.38873, 0., 0., 0.], [0., 0., 0., 0., -1.13389, 1.47842, 0., 0.], [0., 0., 0., 0., 0., -1.24212, 1.56296, 0.], [0., 0., 0., 0., 0., 0., -1.34164, 1.64317]], [[1.06904, 0., 0., 0., 0., 0., 0., 0.], [-0.503953, 1.18187, 0., 0., 0., 0., 0., 0.], [0., -0.712697, 1.28483, 0., 0., 0., 0., 0.], [0., 0., -0.872872, 1.38013, 0., 0., 0., 0.], [0., 0., 0., -1.00791, 1.46926, 0., 0., 0.], [0., 0., 0., 0., -1.12687, 1.55329, 0., 0.], [0., 0., 0., 0., 0., -1.23443, 1.63299, 0.], [0., 0., 0., 0., 0., 0., -1.33333, 1.70899]], [[1.17851, 0., 0., 0., 0., 0., 0., 0.], [-0.502519, 1.28118, 0., 0., 0., 0., 0., 0.], [0., -0.710669, 1.3762, 0., 0., 0., 0., 0.], [0., 0., -0.870388, 1.46508, 0., 0., 0., 0.], [0., 0., 0., -1.00504, 1.54887, 0., 0., 0.], [0., 0., 0., 0., -1.12367, 1.62835, 0., 0.], [0., 0., 0., 0., 0., -1.23091, 1.70412, 0.], [0., 0., 0., 0., 0., 0., -1.32954, 1.77667]]])
    SPlusArray=SPlusArray.astype(complex)
    VArray=np.array([[[-0.752253, -0.71365, -0.674336, -0.642358, -0.61628, -0.594565, \
        -0.576111, -0.560156], [-0.71365, -1.40347, -1.22218, -1.08074, \
        -0.992364, -0.93612, -0.897748, -0.869373], [-0.674336, -1.22218, \
        -1.84721, -1.66056, -1.46245, -1.31971, -1.22419, -1.16037], \
        [-0.642358, -1.08074, -1.66056, -2.21962, -2.03559, -1.80662, \
        -1.62394, -1.49317], [-0.61628, -0.992364, -1.46245, -2.03559, \
        -2.54542, -2.36554, -2.11866, -1.907], [-0.594565, -0.93612, \
        -1.31971, -1.80662, -2.36554, -2.83783, -2.66232, -2.40462], \
        [-0.576111, -0.897748, -1.22419, -1.62394, -2.11866, -2.66232, \
        -3.10496, -2.93366], [-0.560156, -0.869373, -1.16037, -1.49317, \
        -1.907, -2.40462, -2.93366, -3.35218]], [[-1.54101, -0.979665, \
        -0.709766, -0.565628, -0.47891, -0.421029, -0.379071, -0.34673], \
        [-0.979665, -1.87503, -1.4639, -1.14456, -0.936081, -0.799501, \
        -0.70615, -0.638799], [-0.709766, -1.4639, -2.20379, -1.86397, \
        -1.52646, -1.27529, -1.09786, -0.972146], [-0.565628, -1.14456, \
        -1.86397, -2.50873, -2.21109, -1.86757, -1.58805, -1.37842], \
        [-0.47891, -0.936081, -1.52646, -2.21109, -2.79014, -2.52079, \
        -2.17666, -1.87794], [-0.421029, -0.799501, -1.27529, -1.86757, \
        -2.52079, -3.05116, -2.80237, -2.4602], [-0.379071, -0.70615, \
        -1.09786, -1.58805, -2.17666, -2.80237, -3.29493, -3.06191], \
        [-0.34673, -0.638799, -0.972146, -1.37842, -1.87794, -2.4602, \
        -3.06191, -3.52407]], [[-1.98912, -1.16473, -0.750363, -0.529727, \
        -0.402577, -0.323368, -0.27032, -0.232475], [-1.16473, -2.25162, \
        -1.64859, -1.19363, -0.898424, -0.707261, -0.579768, -0.491176], \
        [-0.750363, -1.64859, -2.52093, -2.03276, -1.57436, -1.23727, \
        -1.00006, -0.832966], [-0.529727, -1.19363, -2.03276, -2.78138, \
        -2.36427, -1.91294, -1.55095, -1.27991], [-0.402577, -0.898424, \
        -1.57436, -2.36427, -3.02927, -2.66072, -2.21975, -1.84264], \
        [-0.323368, -0.707261, -1.23727, -1.91294, -2.66072, -3.26446, \
        -2.93134, -2.50148], [-0.27032, -0.579768, -1.00006, -1.55095, \
        -2.21975, -2.93134, -3.48782, -3.18179], [-0.232475, -0.491176, \
        -0.832966, -1.27991, -1.84264, -2.50148, -3.18179, -3.70047]], \
        [[-2.30639, -1.27914, -0.771783, -0.504612, -0.35385, -0.263206, \
        -0.205343, -0.166271], [-1.27914, -2.5513, -1.77966, -1.22098, \
        -0.865687, -0.639701, -0.492387, -0.393094], [-0.771783, -1.77966, \
        -2.7927, -2.16365, -1.60287, -1.20017, -0.921429, -0.728256], \
        [-0.504612, -1.22098, -2.16365, -3.02675, -2.49012, -1.94127, \
        -1.51175, -1.19505], [-0.35385, -0.865687, -1.60287, -2.49012, \
        -3.2519, -2.78036, -2.24762, -1.80292], [-0.263206, -0.639701, \
        -1.20017, -1.94127, -2.78036, -3.46791, -3.04478, -2.52893], \
        [-0.205343, -0.492387, -0.921429, -1.51175, -2.24762, -3.04478, \
        -3.67511, -3.28946], [-0.166271, -0.393094, -0.728256, -1.19505, \
        -1.80292, -2.52893, -3.28946, -3.87407]], [[-2.55498, -1.34775, \
        -0.774978, -0.480505, -0.317789, -0.2224, -0.163458, -0.125198], \
        [-1.34775, -2.79575, -1.86896, -1.22832, -0.831993, -0.585316, \
        -0.427982, -0.324588], [-0.774978, -1.86896, -3.02436, -2.26068, \
        -1.61281, -1.16071, -0.854406, -0.6462], [-0.480505, -1.22832, \
        -2.26068, -3.24353, -2.58906, -1.95292, -1.46903, -1.11951], \
        [-0.317789, -0.831993, -1.61281, -2.58906, -3.45414, -2.87847, \
        -2.26049, -1.75865], [-0.2224, -0.585316, -1.16071, -1.95292, \
        -2.87847, -3.65678, -3.14081, -2.54277], [-0.163458, -0.427982, \
        -0.854406, -1.46903, -2.26049, -3.14081, -3.85198, -3.38286], \
        [-0.125198, -0.324588, -0.6462, -1.11951, -1.75865, -2.54277, \
        -3.38286, -4.04026]]])
    
    VArray=VArray.astype(complex)
    VZArray=(Zeff-1)*np.array([[[-0.752253, -0.71365, -0.674336, -0.642358, -0.61628, -0.594565, \
        -0.576111, -0.560156], [-0.71365, -0.977929, -0.980921, -0.95762, \
        -0.931117, -0.905905, -0.882884, -0.862067], [-0.674336, -0.980921, \
        -1.16331, -1.18126, -1.16978, -1.15031, -1.12894, -1.10779], \
        [-0.642358, -0.95762, -1.18126, -1.32405, -1.34877, -1.34541, \
        -1.33201, -1.31482], [-0.61628, -0.931117, -1.16978, -1.34877, \
        -1.46783, -1.49592, -1.49821, -1.48976], [-0.594565, -0.905905, \
        -1.15031, -1.34541, -1.49592, -1.59904, -1.62884, -1.63519], \
        [-0.576111, -0.882884, -1.12894, -1.33201, -1.49821, -1.62884, \
        -1.72044, -1.75109], [-0.560156, -0.862067, -1.10779, -1.31482, \
        -1.48976, -1.63519, -1.75109, -1.83395]], [[-0.902703, -0.723773, \
        -0.603144, -0.519693, -0.458641, -0.411927, -0.374928, -0.344825], \
        [-0.723773, -1.09614, -0.999422, -0.892889, -0.80356, -0.730611, \
        -0.670595, -0.620533], [-0.603144, -0.999422, -1.26271, -1.20441, \
        -1.11584, -1.03142, -0.956841, -0.892046], [-0.519693, -0.892889, \
        -1.20441, -1.41096, -1.37453, -1.30132, -1.22444, -1.1523], \
        [-0.458641, -0.80356, -1.11584, -1.37453, -1.54575, -1.52316, \
        -1.46236, -1.39326], [-0.411927, -0.730611, -1.03142, -1.30132, \
        -1.52316, -1.67011, -1.65689, -1.60609], [-0.374928, -0.670595, \
        -0.956841, -1.22444, -1.46236, -1.65689, -1.78612, -1.77954], \
        [-0.344825, -0.620533, -0.892046, -1.1523, -1.39326, -1.60609, \
        -1.77954, -1.89523]], [[-1.03166, -0.729494, -0.549877, -0.435829, \
        -0.35807, -0.30209, -0.260083, -0.227525], [-0.729494, -1.2036, \
        -1.01094, -0.836483, -0.703316, -0.601992, -0.523427, -0.461203], \
        [-0.549877, -1.01094, -1.35601, -1.2199, -1.06332, -0.92868, \
        -0.817984, -0.727256], [-0.435829, -0.836483, -1.2199, -1.49413, \
        -1.3927, -1.25433, -1.12467, -1.01166], [-0.35807, -0.703316, \
        -1.06332, -1.3927, -1.62127, -1.54314, -1.42097, -1.29864], \
        [-0.30209, -0.601992, -0.92868, -1.25433, -1.54314, -1.73962, \
        -1.67811, -1.56987], [-0.260083, -0.523427, -0.817984, -1.12467, \
        -1.42097, -1.67811, -1.85076, -1.80161], [-0.227525, -0.461203, \
        -0.727256, -1.01166, -1.29864, -1.56987, -1.80161, -1.95583]], \
        [[-1.14629, -0.73317, -0.508362, -0.375102, -0.289483, -0.231014, \
        -0.189183, -0.158141], [-0.73317, -1.3026, -1.0188, -0.788296, \
        -0.624037, -0.505956, -0.418852, -0.352892], [-0.508362, -1.0188, \
        -1.44388, -1.231, -1.01523, -0.842031, -0.707221, -0.60185], \
        [-0.375102, -0.788296, -1.231, -1.57364, -1.40623, -1.20882, \
        -1.03625, -0.893625], [-0.289483, -0.624037, -1.01523, -1.40623, \
        -1.69423, -1.55846, -1.37891, -1.21135], [-0.231014, -0.505956, \
        -0.842031, -1.20882, -1.55846, -1.80729, -1.69478, -1.53145], \
        [-0.189183, -0.418852, -0.707221, -1.03625, -1.37891, -1.69478, \
        -1.91405, -1.81929], [-0.158141, -0.352892, -0.60185, -0.893625, \
        -1.21135, -1.53145, -1.81929, -2.01544]], [[-1.2505, -0.735729, \
        -0.474911, -0.329162, -0.240287, -0.182395, -0.142726, -0.114435], \
        [-0.735729, -1.39479, -1.02451, -0.746982, -0.560254, -0.4325, \
        -0.342349, -0.276792], [-0.474911, -1.02451, -1.52705, -1.23937, \
        -0.971945, -0.768963, -0.618229, -0.505249], [-0.329162, -0.746982, \
        -1.23937, -1.64977, -1.4167, -1.16619, -0.958841, -0.795015], \
        [-0.240287, -0.560254, -0.971945, -1.4167, -1.76468, -1.57061, \
        -1.33814, -1.13256], [-0.182395, -0.4325, -0.768963, -1.16619, \
        -1.57061, -1.87306, -1.70825, -1.49306], [-0.142726, -0.342349, \
        -0.618229, -0.958841, -1.33814, -1.70825, -1.97588, -1.83379], \
        [-0.114435, -0.276792, -0.505249, -0.795015, -1.13256, -1.49306, \
        -1.83379, -2.0739]]])
    VZArray=VZArray.astype(complex)
    VArray=VArray+VZArray
    L11_grid=np.arange(x_min, x_max+del_x, del_x,dtype=complex)
    L12grid=np.arange(x_min, x_max+del_x, del_x,dtype=complex)
    for i in range(num):
        k= (np.sqrt(2)*x_grid[i]*alpha*np.sqrt(mref*1836.))/v1
        h1 = np.linalg.inv(1j*w_hat*np.identity(8)+VArray[0]-1j*(k**2/w_hat)*np.matmul(SMinusArray[1],SPlusArray[0])+k**2*np.matmul(np.matmul(SPlusArray[1],np.linalg.inv(
                1j*w_hat*np.identity(8)+VArray[1]+ 
                k**2*np.matmul(np.matmul(SPlusArray[2],np.linalg.inv(
                        1j*w_hat*np.identity(8)+VArray[2]+
                        k**2*np.matmul(np.matmul(SPlusArray[3],np.linalg.inv(
                                1j*w_hat*np.identity(8)+VArray[3]+
                                k**2*np.matmul(np.matmul(SPlusArray[4],np.linalg.inv(
                                        1j*w_hat*np.identity(8)+VArray[4])),SMinusArray[5]))),SMinusArray[
                                        4]))),SMinusArray[3]))),SMinusArray[2]))
        L11_grid[i]=np.matmul(h1,np.array([0.5+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0,0,0],dtype=complex))[0]
        L12grid[i]=np.matmul(h1,np.array([0.+0.j,-0.790569+0.j,0,0,0,0,0,0],dtype=complex))[0]
    
    if ModIndex==0:
        ModG=1.
    
    elif ModIndex==1:
        ModG=np.exp(-((x_grid-mu)/xstar)**2)
    else:
        print("ModIndex must be 0 or 1")
        ModG=0
    sigma_grid = (w1*L11_grid-(1.0+eta)*np.multiply(L11_grid,ModG) - eta*np.multiply(L12grid,ModG))/v1
    plt.plot(x_grid,np.real(sigma_grid),label="Re(Sigma)")
    plt.plot(x_grid,np.imag(sigma_grid),label="Im(Sigma)")
    plt.legend()
    plt.show()

def Dispersion(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
    w0=VectorFinder_auto(nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
    return w0


#*******Testing line**************
#VectorFinder_auto(1.5,2.0,2.267679,0.006,0.002,0.05,1,0,6)
#*******Testing line**************

