import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************
n_min=1                                #minmum mode number (include) that finder will cover
n_max=5                              #maximum mode number (include) that finder will cover

running_mode=1      # mode1: fast mode
                    # mode2: slow mode
                    # mode3: slow mode with all the rational surfaces 
q_scale=0.98         #the scaling of the q profile
q_shift=0.

profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
outputpath='./Test_files/'
path='./Test_files/'
profile_name=path+'p174819.03560'
geomfile_name=path+'g174819.03560'

suffix='.dat'
x0_min=0.93         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor
peak_percent=0.03
show_plot=True
output_csv=True
#************End of User Block*****************
#**********************************************

a=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max,\
            show_plot)

Lref=a.Lref     #minor radius
rho_s=a.rhoref  #gyroradius
cs=a.cs         #speed of sound 

cs_to_kHZ=cs/Lref/(2.*np.pi*1000.)

m_list=[]
nu_list=[]
zeff_list=[]
eta_list=[]
shat_list=[]
beta_list=[]
ky_list=[]
ModIndex_list=[]
mu_list=[]
xstar_list=[]

n_list=np.arange(n_min,n_max+1,1)
for n in n_list:
    x_surface_near_peak, m_surface_near_peak=a.Rational_surface_peak_surface(n0):
    nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar=\
        parameter_for_dispersion(x_surface_near_peak)
    index=np.argmin(abs(a.x-x_surface))
    omega_e_kHz=a.ome[index]
    m_list.append(m_surface_near_peak)
    x_list.append(x_surface_near_peak)
    nu_list.append(nu)
    zeff_list.append(zeff)
    eta_list.append(eta)
    shat_list.append(shat)
    beta_list.append(beta)
    ky_list.append(ky)
    ModIndex_list.append(ModIndex)
    mu_list.append(mu)
    xstar_list.append(xstar)

d = {'n':n,'m':m,'rho_tor':x,\
    'omega_n_kHz':omega_n,\
    'cs_to_kHZ':[cs_to_kHZ]*len(n_list),\
    'omega_e_kHz':

    'nu':nu,'zeff':[zeff]*len(n_list),'eta':eta,\
    'shat':shat,'beta':beta,'ky':ky,\
    'ModIndex':ModIndex,'mu':mu,'xstar':xstar}
df=pd.DataFrame(d, columns=['x','y'])   #construct the panda dataframe
df.to_csv(outputpath+'parameter_list.csv',index=False)
    
if running_mode==1:
    pass
else:
    x,data=self.x, self.ome
    amplitude,mean,stddev=a.gaussian_fit_auto(self,x,data)


if running_mode==2:
    w0=a.Dispersion(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
elif running_mode==3:
    w0=a.Dispersion(nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar)

    x_surface_near_peak
if output_csv==True:
    df.tocsv()



