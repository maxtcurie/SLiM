#This script is intended to find the top and the mid pedestal of the H mod plasma profile for the pre and post processing of the simulation

#Developed by Max Curie on 02/03/2020

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import re
from max_stat_tool import *
# some_file.py
import sys


#The command of this script will be "python max_parity_calculator.py 0001(# of the scan)" 

def parity_finder_long(zgrid,f,name,plot,report): #this function is for local linear simulation with long range of z 

    f=np.real(f)
    #For the case the zgrid is evenly distributed. 
    zmin=np.min(zgrid)
    zmax=np.max(zgrid)
    nz=len(zgrid)
    #print(zmax)
    #print(zmin)

    z0=int(nz/(zmax-zmin)) #nz from zmin to zmax
    parity_name=['even','odd']
    location=[-1,0,1]
    parity=np.zeros((len(location)+1,2))
    
    parity_plot_abs0=[]
    parity_plot_abs1=[]
    for j in range(len(location)):
        for i in range(z0):
            ztemp=int(nz/2+location[j]*z0) #around -1,0,1
            #even parity sum(f(x)-f(-x)) = 0 if even 
            parity[j][0]=parity[j][0]+abs(f[ztemp+i-1]-f[ztemp-i])
            #odd parity sum(f(x)+f(-x)) = 0 if odd 
            parity[j][1]=parity[j][1]+abs(f[ztemp+i-1]+f[ztemp-i])


    #********Doing the same thing for loop***********
    z_loop, f_loop=loop(zgrid,f,-1,1)
    for i in range(z0):
        ztemp=z0
        parity_plot_abs0.append(abs(f_loop[ztemp+i-1]-f_loop[ztemp-i]))
        parity_plot_abs1.append(abs(f_loop[ztemp+i-1]+f_loop[ztemp-i]))
        #even parity sum(f(x)-f(-x)) = 0 if even 
        ntemp=len(location)
        parity[ntemp][0]=parity[ntemp][0]+abs(f_loop[ztemp+i-1]-f_loop[ztemp-i])
        #odd parity sum(f(x)+f(-x)) = 0 if odd 
        parity[ntemp][1]=parity[ntemp][1]+abs(f_loop[ztemp+i-1]+f_loop[ztemp-i])
    #***********Find the ratio at different locations****************
    ratio=np.zeros((len(location)+1)*2)
    for i in range(len(location)+1):
        ratio[2*i+0]=parity[i][0]/(parity[i][0]+parity[i][1]) #percentage of oddness
        ratio[2*i+1]=parity[i][1]/(parity[i][0]+parity[i][1]) #percentage of evenness
    
    #ratio[2*0+0,2*0+1]=-1    {percentage of oddness, percentage of eveness}
    #ratio[2*1+0,2*1+1]= 0    {percentage of oddness, percentage of eveness}
    #ratio[2*2+0,2*2+1]= 1    {percentage of oddness, percentage of eveness}
    #ratio[2*3+0,2*3+1]= loop {percentage of oddness, percentage of eveness}
    location0=location[int(np.argmin(ratio[0:len(location)*2-1])/2)]
    parity0=parity_name[int(np.argmin(ratio[0:len(location)*2-1])%2)]


    #********calc the ratio for loop**********************
   
    if parity[3][1]<parity[3][0]:
        parity1='odd'
    else:
        parity1='even'

    #*********Print out report*****************
    if report==1:
        for i in range(len(location)+1):
            if i < len(location):
                print('Around z=',location[i])
            else:
                print('For the loop')        
            print(ratio[2*i+0]*100,'% Odd', ratio[2*i+1]*100, "% Even")

        print('The location of the center is', location0)
        print('The function is largely', parity0)

        print('Based on the loop, the function is largely ',parity1)
    
        if parity1==parity0:
            print('Result checked')
        else:
            print('location=', location0 ,'and loop mismatch, please check the Parity_plot.png or function.png to determine the parity manually')
    #*********End of Print out report**********

    #********Plot the result*******************
    if plot ==1:
        x_zoom,y_zoom = zoom1D(zgrid,f,-2,2)
        plt.clf()
        plt.title('Parity of'+ name+ 'calculation')
        plt.xlabel(r'$z/\pi$',fontsize=10)
        plt.ylabel(r'$f$',fontsize=13)
        #plt.plot(np.arange(0,1,1/len(parity_plot0)),parity_plot_abs0,label="Even function abs")
        #plt.plot(np.arange(0,1,1/len(parity_plot0)),parity_plot_abs1,label="Odd function abs")
        plt.plot(x_zoom,y_zoom,label=name)
        plt.plot(z_loop,f_loop,label=name+'_loop')
        plt.axhline(y=0, color="red")
        plt.axvline(x=location0, label='Center of the symmetry', color="red")
        #plt.axvline(x=midped, label="Mid-pedestal", color="red")
        #lt.axvline(x=topped, label="Top-pedestal", color="green")
        plt.legend()
        plt.savefig('Parity_plot_'+ name +'.png')
        plt.show()

        plt.clf()
        plt.title(name)
        plt.xlabel(r'$z/\pi$',fontsize=10)
        plt.ylabel(r'$f$',fontsize=13)
        #plt.plot(z_loop,f_loop,label="loop_f")
        plt.plot(zgrid,f,label=name)
        #plt.axvline(x=midped, label="Mid-pedestal", color="red")
        #lt.axvline(x=topped, label="Top-pedestal", color="green")
        plt.legend()
        plt.savefig('function_'+name+'.png')
        plt.show()
        
        ntemp=int(np.argmin(ratio[0:len(location)*2-1])/2)
        ratio0=np.zeros(2)
        ratio0[0]=ratio[ntemp*2+0]
        ratio0[1]=ratio[ntemp*2+1]

    return parity0,location0,ratio0

def parity_finder_short(zgrid,f,name,plot,report): #this function is for local linear simulation with short range of z 

    f=np.real(f)
    #For the case the zgrid is evenly distributed. 
    zmin=np.min(zgrid)
    zmax=np.max(zgrid)

    if abs(zmin) < abs(zmax):
        idx=find_nearest_index(zgrid,-zmax)
        new_z=zgrid[idx:-1]
        f=f[idx:-1]
    elif abs(zmin) > abs(zmax):
        idx=find_nearest_index(zgrid,-zmin)
        new_z=zgrid[0:idx]
        f=f[0:idx]
    else:
        new_z=zgrid
        f=f

    idx=find_nearest_index(new_z,0)
    #print(f'index of z=0 is {idx}')
    #print(f'index of nz/2 is {int(len(new_z)/2)}')

    z0=int(len(new_z)/2)

    location0=0
    parity=np.zeros(2)


    for i in range(z0):
        ztemp=int(z0) #around -1,0,1
        #even parity sum(f(x)-f(-x)) = 0 if even 
        parity[0]=parity[0]+abs(f[ztemp+i-1]-f[ztemp-i])
        #odd parity sum(f(x)+f(-x)) = 0 if odd 
        parity[1]=parity[1]+abs(f[ztemp+i-1]+f[ztemp-i])

    #Determine function's parity
    if parity[1]<parity[0]:
        parity1='odd'
    else:
        parity1='even'

    if plot ==1:
        plt.clf()
        plt.title('Parity of'+name+' calculation')
        plt.xlabel(r'$z/\pi$',fontsize=10)
        plt.ylabel(name,fontsize=13)
        #plt.plot(np.arange(0,1,1/len(parity_plot0)),parity_plot_abs0,label="Even function abs")
        #plt.plot(np.arange(0,1,1/len(parity_plot0)),parity_plot_abs1,label="Odd function abs")
        plt.plot(new_z,f,label=name)
        plt.axhline(y=0, color="red")
        plt.axvline(x=location0, label='Center of the symmetry', color="red")
        #plt.axvline(x=midped, label="Mid-pedestal", color="red")
        #lt.axvline(x=topped, label="Top-pedestal", color="green")
        plt.legend()
        plt.savefig('Parity_plot_'+name+'.png')
        plt.show()

    ratio=np.zeros(2)
    ratio[0]=parity[0]/(parity[0]+parity[1]) #percentage of oddness
    ratio[1]=parity[1]/(parity[0]+parity[1]) #percentage of evenness

    return parity1,location0,ratio

def parity_finder_general(zgrid,f,name,plot,report):
    zmin=np.min(zgrid)
    zmax=np.max(zgrid)
    if zmax>=2 and zmin<= -2: 
        return parity_finder_long(zgrid,f,name,plot,report)
    else:
        return parity_finder_short(zgrid,f,name,plot,report)