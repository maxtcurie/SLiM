import numpy as np

from Tools.DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto
from Tools.DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
from Tools.DispersionRelationDeterminantFullConductivityZeff import VectorFinder_initial_guess
from SLiM_NN.Dispersion_NN import Dispersion_NN

#************Start of user block******************
run_mode=0      # mode0: input_initial_guess(global, 100sec/mode)
                # mode1: slow mode manual(global, 20sec/mode)
                # mode2: slow slow mode(global, 100sec/mode)
                # mode3: NN mode (global, 0.05sec/mode)

nu=1.
zeff=1.
eta=2.
shat=0.006
beta=0.002
ky=0.01
mu=0.
xstar=10

NN_path='./SLiM_NN/Trained_model_backup_03_31_2022/'

#************End of user block******************

if run_mode==0:
    w=VectorFinder_initial_guess(nu,zeff,eta,shat,beta,ky,1,mu,xstar,eta+1.1+0.4j)
elif run_mode==1:
    w=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,1,mu,xstar)  
elif run_mode==2:
    w=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,1,mu,xstar) 
elif run_mode==3:
    Dispersion_NN_obj=Dispersion_NN(NN_path)
    w=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar)
    

print('******w*****')
print('w='+str(w))