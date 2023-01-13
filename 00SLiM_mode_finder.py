import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm  #for progress bar

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************

#for signal q
q_scale_list=[1.]
q_shift_list=[0.]
ne_scale_list=[1.]
ne_shift_list=[0.]
te_scale_list=[1.]
te_shift_list=[0.]


n_min=1         #minmum mode number (include) that finder will cover
n_max=5         #maximum mode number (include) that finder will cover
surface_num=2   #total amount of rational surfaces one want to look at

Run_mode=2      # mode1: fast mode
                # mode2: slow mode(global)
                # mode3: slow mode(local) 
                # mode4: slow mode manual(global)
                # mode5: slow slow mode(global)
                # mode6: NN mode (global)

profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
Output_Path='./Output/'
Output_suffix='_3'
InputPath='./Test_files/'
profile_name=InputPath+'p174819.03560'
geomfile_name=InputPath+'g174819.03560'

manual_fit=False #Change to False for auto fit for xstar

zeff_manual=-1     #set to -1 automatically calculate the zeff

suffix='.dat'       #for the 'GENE_tracor' geomfile
x0_min=0.93         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor
peak_percent=20.
mref=2.             #mess of ion in unit of proton, for deuterium is 2
Impurity_charge=6.  #charge of impurity, for carbon is 6
show_plot=False


NN_path='./SLiM_NN/Trained_model/'

#************End of User Block*****************
#**********************************************

scale_list=[
        [q_scale,q_shift,\
        ne_scale,ne_shift,\
        te_scale,te_shift]\
    \
    for     q_scale,q_shift,\
            ne_scale,ne_shift,\
            te_scale,te_shift \
    \
    in zip( q_scale_list,q_shift_list,\
            ne_scale_list,ne_shift_list,\
            te_scale_list,te_shift_list)
    ]


mode_finder_obj=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            Output_Path,InputPath,x0_min,x0_max,\
            zeff_manual,suffix,\
            mref,Impurity_charge)
x_list=[]
n_list=[]
m_list=[]
nu_list=[]
zeff_list=[]
eta_list=[]
shat_list=[]
beta_list=[]
ky_list=[]
ModIndex_list=[]
mu_list=[]
omega_e_plasma_list=[]
omega_e_lab_list=[]
omega_n_kHz_list=[]
omega_n_cs_a_list=[]
xstar_list=[]
q_scale_list0=[]
q_shift_list0=[]

for i in range(len(scale_list)):
    [   q_scale,q_shift,\
        ne_scale,ne_shift,\
        te_scale,te_shift\
        ]=scale_list[i]

    mode_finder_obj.reset_profile()
    mode_finder_obj.modify_profile(q_scale=q_scale,q_shift=q_shift,\
                                    ne_scale=ne_scale,te_scale=te_scale,\
                                    ne_shift=ne_shift,te_shift=te_shift,\
                                    Doppler_scale=1.,\
                                    show_plot=True)
    mode_finder_obj.ome_peak_range(peak_percent)
    mean_rho,xstar=mode_finder_obj.omega_gaussian_fit(manual=manual_fit)
    mode_finder_obj.set_xstar(xstar)
    
    if Run_mode==1:#simple rational surface alignment
        ModIndex=-1
        filename='rational_surface_alignment'+Output_suffix+'.csv'
    if Run_mode==2 or Run_mode==4 or Run_mode==5 or Run_mode==6:#global dispersion
        ModIndex=1
        filename='global_dispersion'+Output_suffix+'.csv'
    elif Run_mode==3:#local dispersion
        ModIndex=0
        filename='local_dispersion'+Output_suffix+'.csv'
    
    
    peak_index=np.argmin(abs(mode_finder_obj.x-mode_finder_obj.x_peak))
    omega_e_peak_kHz=mode_finder_obj.ome[peak_index]
    
    cs_to_kHz=mode_finder_obj.cs_to_kHz[peak_index]
    print('Finding the rational surfaces')
    n0_list=np.arange(n_min,n_max+1,1)

    for n in tqdm(n0_list):
        x_surface_near_peak_list, m_surface_near_peak_list=mode_finder_obj.Rational_surface_top_surfaces(n,top=surface_num)
        print(x_surface_near_peak_list)
        print(m_surface_near_peak_list)
        for j in range(len(x_surface_near_peak_list)):
            x_surface_near_peak=x_surface_near_peak_list[j]
            m_surface_near_peak=m_surface_near_peak_list[j]
            if mode_finder_obj.x_min<=x_surface_near_peak and x_surface_near_peak<=mode_finder_obj.x_max:
                nu,zeff,eta,shat,beta,ky,mu,xstar=\
                    mode_finder_obj.parameter_for_dispersion(x_surface_near_peak,n)
                factor=mode_finder_obj.factor
                index=np.argmin(abs(mode_finder_obj.x-x_surface_near_peak))
                omega_n_kHz=float(n)*mode_finder_obj.omn[index]
                omega_n_cs_a=float(n)*mode_finder_obj.omn[index]/cs_to_kHz
                omega_e_plasma_kHz=float(n)*mode_finder_obj.ome[index]
                omega_e_lab_kHz=float(n)*mode_finder_obj.ome[index]+float(n)*mode_finder_obj.Doppler[index]
            
                n_list.append(n)
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
                omega_e_plasma_list.append(omega_e_plasma_kHz)
                omega_e_lab_list.append(omega_e_lab_kHz)
                omega_n_kHz_list.append(omega_n_kHz)
                omega_n_cs_a_list.append(omega_n_cs_a)
                q_scale_list0.append(q_scale)
                q_shift_list0.append(q_shift)
    
    
d = {'q_scale':q_scale_list0,'q_shift':q_shift_list0,\
    'n':n_list,'m':m_list,'rho_tor':x_list,\
    'omega_plasma_kHz':[0]*len(n_list),\
    'omega_lan_kHz':[0]*len(n_list),\
    'gamma_cs_a':[0]*len(n_list),\
    'omega_n_kHz':omega_n_kHz_list,\
    'omega_n_cs_a':omega_n_cs_a_list,\
    'omega_e_plasma_kHz':omega_e_plasma_list,\
    'omega_e_lab_kHz':omega_e_lab_list,\
    'peak_percentage':omega_e_plasma_list/\
            (omega_e_peak_kHz*np.array(n_list,dtype=float)),\
    'nu':nu_list,'zeff':[zeff]*len(n_list),'eta':eta_list,\
    'shat':shat_list,'beta':beta_list,'ky':ky_list,\
    'ModIndex':ModIndex_list,'mu':mu_list,'xstar':xstar_list}
df=pd.DataFrame(d, columns=['q_scale','q_shift','n','m','rho_tor',\
    'omega_plasma_kHz','omega_lab_kHz','gamma_cs_a','omega_n_kHz',\
    'omega_n_cs_a','omega_e_plasma_kHz','omega_e_lab_kHz',\
    'peak_percentage','nu','zeff','eta','shat','beta','ky',\
    'ModIndex','mu','xstar'])   #construct the panda dataframe
df.to_csv(Output_Path+'parameter_list'+Output_suffix+'.csv',index=False)

if Run_mode==6:
    from SLiM_NN.Dispersion_NN import Dispersion_NN
    Dispersion_NN_obj=Dispersion_NN(NN_path)

    
if Run_mode==1:
    pass
else:    
    with open(Output_Path+filename, 'w', newline='') as csvfile:     #clear all and then write a row
        data = csv.writer(csvfile, delimiter=',')
        data.writerow(['q_scale','q_shift','n','m','rho_tor',\
            'omega_plasma_kHz','omega_lab_kHz',\
            'gamma_cs_a','omega_n_kHz',\
            'omega_n_cs_a','omega_e_plasma_kHz',\
            'omega_e_lab_kHz','peak_percentage',\
            'nu','zeff','eta','shat','beta','ky',\
            'ModIndex','mu','xstar'])
    csvfile.close()

    print('Calculate the dispersion relations')
    
    for i in tqdm(range(len(n_list))):
        if Run_mode==4:
            w0=mode_finder_obj.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                df['shat'][i],df['beta'][i],df['ky'][i],\
                df['ModIndex'][i],df['mu'][i],df['xstar'][i],manual=True)
        elif Run_mode==5:
            w0=mode_finder_obj.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                df['shat'][i],df['beta'][i],df['ky'][i],\
                df['ModIndex'][i],df['mu'][i],df['xstar'][i],manual=5)
        elif Run_mode==6:
            w0=Dispersion_NN_obj.Dispersion_omega(df['nu'][i],df['zeff'][i],df['eta'][i],\
                    df['shat'][i],df['beta'][i],df['ky'][i],df['mu'][i],df['xstar'][i])
        else:
            w0=mode_finder_obj.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                df['shat'][i],df['beta'][i],df['ky'][i],\
                df['ModIndex'][i],df['mu'][i],df['xstar'][i])
        
        omega=np.real(w0)
        omega_kHz=omega*omega_n_kHz_list[i]
        gamma=np.imag(w0)
        gamma_cs_a=gamma*omega_n_cs_a_list[i]
        with open(Output_Path+filename, 'a+', newline='') as csvfile: #adding a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([ q_scale_list0[i],\
                q_shift_list0[i],\
                df['n'][i],df['m'][i],df['rho_tor'][i],\
                omega_kHz,\
                omega_kHz+df['omega_e_lab_kHz'][i]-df['omega_e_plasma_kHz'][i],\
                gamma_cs_a,\
                df['omega_n_kHz'][i],df['omega_n_cs_a'][i],\
                df['omega_e_plasma_kHz'][i],df['omega_e_lab_kHz'][i],
                df['peak_percentage'][i],df['nu'][i],\
                df['zeff'][i],df['eta'][i],\
                df['shat'][i],df['beta'][i],df['ky'][i],\
                df['ModIndex'][i],df['mu'][i],df['xstar'][i] ])
        csvfile.close()

