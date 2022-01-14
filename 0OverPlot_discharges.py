import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************

path_list =['./../Discharge_survey/D3D/169510/1966/',\
            './../Discharge_survey/D3D/169510/3069/',\
            './../Discharge_survey/D3D/169510/4069/']

profile_name_list =[path_list[0]+'DIIID169510.01966_108.iterdb',\
                    path_list[1]+'DIIID169510.03090_595.iterdb',\
                    path_list[2]+'DIIID169510.04069_173.iterdb']
geomfile_name_list=[path_list[0]+'g169510.01966_108',\
                    path_list[1]+'g169510.03090_595',
                    path_list[2]+'g169510.04069_173']

profile_type= "ITERDB"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"

outputpath='./Test_files/'


suffix='.dat'
x0_min=0.93         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor
peak_percent=0.01
order=10            # order of polynomial for q profile fit 

marker_size=5
font_size=15
q_uncertainty=0.1
mref=2.
profile_uncertainty=0.2
doppler_uncertainty=0.5

color_list=['orange','green','red']#list of color for the mode number
name_list=['t=1966ms','t=3090ms','t=4069ms']
#************End of User Block*****************
#**********************************************

a_list=[]
plt.clf()
for i in range(len(profile_name_list)):
    profile_name=profile_name_list[i]
    geomfile_name=geomfile_name_list[i]
    path=path_list[i]
    a=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)
    a_list.append(a)
    plt.plot(a.x,a.ome,label=name_list[i],color=color_list[i])
plt.xlabel(r'$\rho_{tor}$')
plt.ylabel(r'$\omega_{*e}$(kHz)')
plt.legend()
plt.show()


n_list=[1,2,3,4,5]
fig, ax=plt.subplots(nrows=len(n_list),\
    ncols=len(a_list),sharex=True,sharey=True) 
(nr,nc)=np.shape(ax)
for i in range(len(n_list)):
    n=n_list[i]
    for j in range(len(a_list)):
        a=a_list[j]
        x_surface_near_peak_list, \
            m_surface_near_peak_list\
            =a.Rational_surface_top_surfaces(n,top=3)
        ax[i,j].plot(a.x,a.ome,label=r'$\omega_{*e}$')

        for x in x_surface_near_peak_list:
            ax[i,j].axvline(x,color='red',label='Ratinoal surface')

for i in range(nr):
    for j in range(nc):
        #if i!=nr-1:
        #    ax[i,j].set_xticklabels([])
        if i==nr-1:
            ax[i,j].set_xlabel(r'$\rho_{tor}$')
        if i==0:
            ax[i,j].set_title(name_list[j])
        ax[i,j].set_yticklabels([])
        if j==0:
            ax[i,j].set_ylabel('n='+str(n_list[i]))
ax[0,nc-1].legend()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()  


n_list=[1,2,3,4,5]
fig, ax=plt.subplots(nrows=len(n_list),\
    ncols=len(a_list),sharex=True,sharey=True) 
(nr,nc)=np.shape(ax)
for i in range(len(n_list)):
    n=n_list[i]
    for j in range(len(a_list)):
        a=a_list[j]
        x_surface_near_peak_list, \
            m_surface_near_peak_list\
            =a.Rational_surface_top_surfaces(n,top=3)
        ax[i,j].plot(a.x,a.ome,label=r'$\omega_{*e}$')
        ax[i,j].plot(a.x,a.ome+a.Doppler,label=r'$\omega_{*e}$')

        for x in x_surface_near_peak_list:
            ax[i,j].axvline(x,color='red',label='Ratinoal surface')
        ax[i,j].grid()


for i in range(nr):
    for j in range(nc):
        #if i!=nr-1:
        #    ax[i,j].set_xticklabels([])
        if i==nr-1:
            ax[i,j].set_xlabel(r'$\rho_{tor}$')
        if i==0:
            ax[i,j].set_title(name_list[j])
        if j==0:
            ax[i,j].set_ylabel('n='+str(n_list[i]))
ax[0,nc-1].legend()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()  