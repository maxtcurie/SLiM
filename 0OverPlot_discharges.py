import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************

path_list =['./Test_files/DIII_D_169510/1966/',\
            './Test_files/DIII_D_169510/3069/',\
            './Test_files/DIII_D_169510/4069/']

profile_name_list =[path_list[0]+'DIIID169510.01966_108.iterdb',\
                    path_list[1]+'DIIID169510.03090_595.iterdb',\
                    path_list[2]+'DIIID169510.04069_173.iterdb']
geomfile_name_list=[path_list[0]+'g169510.01966_108',\
                    path_list[1]+'g169510.03090_595',
                    path_list[2]+'g169510.04069_173']

profile_type= "ITERDB"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"

n_list=[1,2,3,4,5]  #list of toroidal mode numbers
rational_surface_number=3 

color_list=['orange','green','red']#list of color for the mode number
name_list=['t=1966ms','t=3090ms','t=4069ms']

plot_ome=True
plot_ome_surface_clean=True
plot_ome_surface_full=True
plot_parameter=True

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
#************End of User Block*****************
#**********************************************

a_list=[]
for i in range(len(profile_name_list)):
    profile_name=profile_name_list[i]
    geomfile_name=geomfile_name_list[i]
    path=path_list[i]
    a=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)
    a_list.append(a)

if plot_ome==True:
    plt.clf()
    for i in range(len(profile_name_list)):
        a=a_list[i]
        plt.plot(a.x,a.ome,label=name_list[i],color=color_list[i])
    plt.xlabel(r'$\rho_{tor}$')
    plt.ylabel(r'$\omega_{*e}$(kHz)')
    plt.legend()
    plt.show()

if plot_ome_surface_clean==True:
    fig, ax=plt.subplots(nrows=len(n_list),\
        ncols=len(a_list),sharex=True,sharey=True) 
    (nr,nc)=np.shape(ax)
    f_lab_max=np.zeros((nr,nc))
    for i in range(len(n_list)):
        n=n_list[i]
        for j in range(len(a_list)):
            a=a_list[j]
            x_surface_near_peak_list, \
                m_surface_near_peak_list\
                =a.Rational_surface_top_surfaces(n,top=rational_surface_number)
            ax[i,j].plot(a.x,a.ome,label=r'$\omega_{*e}$')
            f_lab_max[i,j]=np.max(float(n)*(a.ome+a.Doppler))
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

if plot_ome_surface_full==True:
    fig, ax=plt.subplots(nrows=len(n_list),\
        ncols=len(a_list),sharex=True) 
    (nr,nc)=np.shape(ax)
    f_lab_max=np.zeros((nr,nc))
    for i in range(len(n_list)):
        n=n_list[i]
        for j in range(len(a_list)):
            a=a_list[j]
            x_surface_near_peak_list, \
                m_surface_near_peak_list\
                =a.Rational_surface_top_surfaces(n,top=rational_surface_number)
            ax[i,j].plot(a.x,float(n)*a.ome,label=r'$\omega_{*e}(plasma)$')
            ax[i,j].plot(a.x,float(n)*(a.ome+a.Doppler),label=r'$\omega_{*e}(lab)$')
            f_lab_max[i,j]=np.max(float(n)*(a.ome+a.Doppler))
            for x in x_surface_near_peak_list:
                ax[i,j].axvline(x,color='red',label='Ratinoal surface')
            #ax[i,j].grid()
    
    
    for i in range(nr):
        for j in range(nc):
            ax[i,j].set_ylim(0, np.max(f_lab_max[i,:])*1.2 ) 
            #if i!=nr-1:
            #    ax[i,j].set_xticklabels([])
            if i==nr-1:
                ax[i,j].set_xlabel(r'$\rho_{tor}$')
            if i==0:
                ax[i,j].set_title(name_list[j])
            if j!=0:
                ax[i,j].set_yticklabels([])
            if j==0:
                ax[i,j].set_ylabel('n='+str(n_list[i]))
    ax[0,nc-1].legend()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()  

if plot_parameter==True:
    fig, ax=plt.subplots(nrows=7,\
        ncols=len(a_list),sharex=True) 
    (nr,nc)=np.shape(ax)
    
    shat_list=[]
    eta_list=[]
    ky_list=[]
    nu_list=[]
    beta_list=[]
    q_list=[]
    ome_list=[]
    for i in range(len(a_list)):
        a=a_list[i]
        ax[0,i].plot(a.x,a.shat)
        ax[0,0].set_ylabel(r'$L_{ne}/L_{q}$')
        shat_list.append(np.min(a.shat))
        shat_list.append(np.max(a.shat))

        ax[1,i].plot(a.x,a.eta)
        ax[1,0].set_ylabel(r'$L_{ne}/L_{Te}$')
        eta_list.append(np.min(a.eta))
        eta_list.append(np.max(a.eta))

        ax[2,i].plot(a.x,a.ky)
        ax[2,0].set_ylabel(r'$k_y\rho_s$')
        ky_list.append(np.min(a.ky))
        ky_list.append(np.max(a.ky))

        ax[3,i].plot(a.x,a.nu)
        ax[3,0].set_ylabel(r'$\nu_{ei}/\omega_{*ne}$')
        nu_list.append(np.min(a.nu))
        nu_list.append(np.max(a.nu))

        ax[4,i].plot(a.x,a.beta)
        ax[4,0].set_ylabel(r'$\beta$')
        beta_list.append(np.min(a.beta))
        beta_list.append(np.max(a.beta))

        ax[5,i].plot(a.x,a.q,label=r'$q$')
        ax[5,0].set_ylabel(r'$q$')
        q_list.append(np.min(a.q))
        q_list.append(np.max(a.q))

        ax[6,i].plot(a.x,a.ome)
        ax[6,0].set_ylabel(r'$\omega_{*e}(kHz)$')
        ome_list.append(np.min(a.ome))
        ome_list.append(np.max(a.ome))

    for i in range(len(a_list)):
        ax[0,i].set_ylim(np.min(shat_list)*0.8,np.max(shat_list)*1.2)
        ax[1,i].set_ylim(np.min(eta_list)*0.8,np.max(eta_list)*1.2)
        ax[2,i].set_ylim(np.min(ky_list)*0.8,np.max(ky_list)*1.2)
        ax[3,i].set_ylim(np.min(nu_list)*0.8,np.max(nu_list)*1.2)
        ax[4,i].set_ylim(np.min(beta_list)*0.8,np.max(beta_list)*1.2)
        ax[5,i].set_ylim(np.min(q_list)*0.8,np.max(q_list)*1.2)
        ax[6,i].set_ylim(np.min(ome_list)*0.8,np.max(ome_list)*1.2)
    
    for i in range(nr):
        for j in range(nc):
            #if i!=nr-1:
            #    ax[i,j].set_xticklabels([])
            if i==nr-1:
                ax[i,j].set_xlabel(r'$\rho_{tor}$')
            if i==0:
                ax[i,j].set_title(name_list[j])
            if j!=0:
                ax[i,j].set_yticklabels([])
                
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()  