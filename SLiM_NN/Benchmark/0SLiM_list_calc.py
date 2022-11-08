import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(1, './../../Tools')
sys.path.insert(1, './../')

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
from Dispersion_NN import Dispersion_NN_beta
from Dispersion_NN import Dispersion_NN_beta_2
#************Start of user block******************

output_csv_file='./SLiM_calc_NN.csv'

run_mode=3      # mode1: slow mode manual(global, 20sec/mode)
                # mode2: slow slow mode(global, 100sec/mode)
                # mode3: NN mode (global, 0.05sec/mode)

mode=1
     
#for n=3, 1.03q0, DIIID 
nu=1.398957621
zeff=2.788549247
eta=1.158925479
shat=0.005906846
beta=0.000710695
ky=0.040731854
mu=0.
xstar=10.72829394

eta_list=np.arange(0.5,4,0.1)

para_list=[[nu, zeff, eta, shat,  beta,  ky,   mu, xstar]\
            for eta in eta_list]



if mode ==1:
    path_tmp='./../Trained_model/'
    NN_stability_file  =path_tmp+'SLiM_NN_stability.h5'
    NN_omega_file      =path_tmp+'SLiM_NN_omega.h5'
    NN_gamma_file      =path_tmp+'SLiM_NN_gamma.h5'

    norm_stability_csv_file=path_tmp+'NN_stability_norm_factor.csv'
    norm_omega_csv_file    =path_tmp+'NN_omega_norm_factor.csv'
    norm_gamma_csv_file    =path_tmp+'NN_gamma_norm_factor.csv'
elif mode==2:
    path_tmp='./../Trained_model/'
    NN_file  =path_tmp+'SLiM_NN.h5'

    norm_csv_file=path_tmp+'NN_norm_factor.csv'


#************End of user block******************


if run_mode==3:
    if mode ==1:
        Dispersion_NN_obj=Dispersion_NN_beta(NN_stability_file,\
                                         NN_omega_file,\
                                         NN_gamma_file,\
                                        norm_stability_csv_file,\
                                        norm_omega_csv_file,\
                                        norm_gamma_csv_file)
    elif mode ==2:
        Dispersion_NN_obj=Dispersion_NN_beta_2(NN_file,norm_csv_file)

f_list=[]
gamma_list=[]
gamma_10_list=[]

nu_output_list=[]
zeff_output_list=[]
eta_output_list=[]
shat_output_list=[]
beta_output_list=[]
ky_output_list=[]
mu_output_list=[]
xstar_output_list=[]

for para in para_list:
    [nu, zeff, eta, shat,  beta,  ky,   mu, xstar]=para
    nu_output_list.append(nu)
    zeff_output_list.append(zeff)
    eta_output_list.append(eta)
    shat_output_list.append(shat)
    beta_output_list.append(beta)
    ky_output_list.append(ky)
    mu_output_list.append(mu)
    xstar_output_list.append(xstar)
    if run_mode==1:
        w=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,1,mu,xstar)  
    elif run_mode==2:
        w=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,1,mu,xstar) 
    elif run_mode==3:
        w=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar)
    
    print(w)

    #if w.imag>0.9:
    #     gamma_10_list.append(1)
    #else:
    #     gamma_10_list.append(0)

    f_list.append(w.real)
    gamma_list.append(w.imag)
    
d = {'nu':nu_output_list, \
    'zeff':zeff_output_list,\
    'eta':eta_output_list,\
    'shat':shat_output_list,\
    'beta':beta_output_list, \
    'ky':ky_output_list,  \
    'mu':mu_output_list, \
    'xstar':xstar_output_list,\
    'f':f_list,'gamma':gamma_list}
    #,'gamma_10':gamma_10_list}
df=pd.DataFrame(d)  #construct the panda dataframe
df.to_csv(output_csv_file,index=False)