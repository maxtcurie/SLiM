import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(1, './../../../Tools')
sys.path.insert(1, './../../')

from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
from Dispersion_NN import Dispersion_NN

#************Start of user block******************

output_csv_file='./SLiM_calc_NN.csv'

run_mode=3      # mode1: slow mode manual(global, 20sec/mode)
                # mode2: slow slow mode(global, 100sec/mode)
                # mode3: NN mode (global, 0.05sec/mode)


     
#for n=3, 1.03q0, DIIID 
nu=1.398957621
zeff=2.788549247
eta=1.158925479
shat=0.005906846
beta=0.000710695
ky=0.040731854
mu=0.
xstar=10.72829394

mu_list=np.arange(0,10,0.1)

para_list=[[nu, zeff, eta, shat,  beta,  ky,   mu, xstar]\
            for mu in mu_list]

path='./../../Trained_model/'
NN_omega_file      =path+'SLiM_NN_omega.h5'
NN_gamma_file      =path+'SLiM_NN_stabel_unstable.h5'
norm_omega_csv_file=path+'NN_omega_norm_factor.csv'
norm_gamma_csv_file=path+'NN_stabel_unstable_norm_factor.csv'

#************End of user block******************


if run_mode==3:
    Dispersion_NN_obj=Dispersion_NN(NN_omega_file,NN_gamma_file,norm_omega_csv_file,norm_gamma_csv_file)

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

for para in tqdm(para_list):
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
    
    if w.imag>0.9:
         gamma_10_list.append(1)
    else:
         gamma_10_list.append(0)

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
    'f':f_list,'gamma':gamma_list,'gamma_10':gamma_10_list}
df=pd.DataFrame(d)  #construct the panda dataframe
df.to_csv(output_csv_file,index=False)