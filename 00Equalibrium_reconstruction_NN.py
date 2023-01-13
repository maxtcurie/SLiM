import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os 

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************
#**********************************************
#**********Start of User block*****************
Frequency_list=[65,110] #frequency observed from experiment
weight_list=[1.,1.3]    #weight mu calculation for each frequency
Frequency_error=0.10    #error for frequency
q_scan_f_err   =0.2    #error for frequency when doing q scan
q_scale_list=np.arange(0.95,1.051,0.01)
q_shift_list=np.array([0.],dtype=float)
shat_scale_list=np.arange(0.95,1.051,0.01)
ne_scale_list=np.arange(0.8,1.21,0.05)
te_scale_list=np.arange(0.8,1.21,0.05)
ne_shift_list=np.array([0.],dtype=float) #default: [0.]
te_shift_list=np.array([0.],dtype=float) #default: [0.]
Doppler_shift_scale_list=np.arange(0.5,1.5,0.1)

scan_mode=0     #scan_mode=-1, take 1 q scaling and do ne, te scan(stop if one matches): 10s
                #scan_mode=0, take 1 q scaling and do ne, te scan: 10s
                #scan_mode=1, take all working q scale and do ne, te scan: 30min
                #scan_mode=2, take all q scale and do ne, te scan: 11hr

reject_band_outside_for_q_scan=False #Change to True if one want to the q scale \
                                    #  that only have the desired frequency band

reject_band_outside_for_scale_scan=False #Change to True if one want to the q scale \
                                            #  that only have the desired frequency band

profile_type= "ITERDB"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
Output_Path='./Output/Equalibrium/'
Output_suffix='_3'
InputPath='./Test_files/'
profile_name=InputPath+'DIIID174819.iterdb'
geomfile_name=InputPath+'g174819.03560'

manual_fit=False #Change to False for auto fit for xstar
surface_num=2

zeff_manual=-1      #set to -1 automatically calculate the zeff

suffix='.dat'       #for the 'GENE_tracor' geomfile
x0_min=0.6         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor
peak_percent=0.08
mref=2.             #mess of ion in unit of proton, for deuterium is 2
Impurity_charge=6.  #charge of impurity, for carbon is 6
show_plot=True

path_tmp='./SLiM_NN/Trained_model/'
NN_omega_file      =path_tmp+'SLiM_NN_omega.h5'
NN_gamma_file      =path_tmp+'SLiM_NN_stabel_unstable.h5'
norm_omega_csv_file=path_tmp+'NN_omega_norm_factor.csv'
norm_gamma_csv_file=path_tmp+'NN_stabel_unstable_norm_factor.csv'
#************End of User Block*****************
#**********************************************



def total_score_calc(best_f_error_avg,n_work_list,df_scale):

    q_scale_weight=2.   #weight for the scale score
    q_shift_weight=2.
    shat_scale_weight=2.
    ne_scale_weight=1.
    ne_shift_weight=1.
    te_scale_weight=1.
    te_shift_weight=1.
    Doppler_shift_scale_weight=0.5

    scale_weight=[  q_scale_weight,
                    q_shift_weight,
                    shat_scale_weight,
                    ne_scale_weight,
                    ne_shift_weight,
                    te_scale_weight,
                    te_shift_weight,
                    Doppler_shift_scale_weight
                    ]


    scale_weight=np.array(scale_weight)
    df_scale=np.array(df_scale)
    scale_score=np.sum(abs(1.-df_scale)*scale_weight)
    scale_score=scale_score/np.sum(scale_weight)

    total_score=(best_f_error_avg+scale_score) #*(1.-np.prod(n_work_list))

    return scale_score,total_score
    
#ignore warning
if ignore_warning:
    #https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    import warnings
    warnings.filterwarnings("ignore")

if not os.path.exists(Output_Path):
    os.makedirs(Output_Path)
else:
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(Output_Path) if (isfile(join(Output_Path, f)) and f!='.gitignore')]
    for i in onlyfiles:
        os.remove(join(Output_Path, i))



mode_finder_obj=mode_finder(profile_type,profile_name,\
                            geomfile_type,geomfile_name,\
                            Output_Path,InputPath,x0_min,x0_max,\
                            zeff_manual,suffix,\
                            mref,Impurity_charge)
index=np.argmax(mode_finder_obj.ome)
                
omega_e_lab_kHz=mode_finder_obj.ome[index]+mode_finder_obj.Doppler[index]

#n_min=1                              #minmum mode number (include) that finder will cover
#n_max=int((1+Frequency_error)*np.max(Frequency_list)/omega_e_lab_kHz)#maximum mode number (include) that finder will cover

#n0_list=np.arange(n_min,n_max+1,1)
n0_list=n_list_observe
n_min=np.min(n_list_observe)                            
n_max=np.max(n_list_observe)
x_peak,x_min,x_max=mode_finder_obj.ome_peak_range(peak_percent=0.1)


peak_index=np.argmin(abs(mode_finder_obj.x-mode_finder_obj.x_peak))
omega_e_peak_kHz=mode_finder_obj.ome[peak_index]

cs_to_kHz=mode_finder_obj.cs_to_kHz[peak_index]


with open(Output_Path+'0Equalibrium_summary.csv', 'w', newline='') as csvfile:     #clear all and then write a row
    data = csv.writer(csvfile, delimiter=',')
    data.writerow(['q_scale','q_shift',\
                'shat_scale',\
                'ne_scale','ne_shift',\
                'te_scale','te_shift',\
                'Doppler_shift_scale',\
                'n_list','m_list',\
                'frequency_match',\
                'n_num_match',\
                'frequency_list',\
                'frequency_error_list',\
                'best_frequency_list',\
                'best_frequency_error_list',\
                'best_f_error_avg',\
                'more_f_band_than_need',\
                'scale_score',\
                'total_score',\
                'file_name'])
csvfile.close()


#***step 1:
freq_min_list=[f*(1.-q_scan_f_err) for f in Frequency_list]
freq_max_list=[f*(1.+q_scan_f_err) for f in Frequency_list]

#find the q profile for the frequency
mu_works_list=[] #the list of mu that has frequency observed from experiment
q_scale_works_list=[]#the list of q_scale that has frequency observed from experiment
q_shift_works_list=[]#the list of q_shift that has frequency observed from experiment
shat_scale_works_list=[]

parameter_list=[[q_scale,q_shift,shat_scale]\
                for q_scale in q_scale_list\
                for q_shift in q_shift_list\
                for shat_scale in shat_scale_list\
                ]
print('********q_scan********')

for [q_scale,q_shift,shat_scale] in tqdm(parameter_list):

    judge_list=np.zeros(len(Frequency_list),dtype=int)

    mode_finder_obj.q_back_to_nominal()
    mode_finder_obj.q_modify(q_scale,q_shift,shat_scale)
    mode_finder_obj.ome_peak_range(peak_percent)

    omega_e_plasma_kHz_list=[]
    omega_e_lab_kHz_list=[]
    Doppler_kHz_list=[]
    mu_list_temp=[]
    outside_frequency=[]
    for n in n0_list:
        x_surface_near_peak_list, m_surface_near_peak_list=mode_finder_obj.Rational_surface_top_surfaces(n,top=3)
        #print(x_surface_near_peak_list)
        #print(m_surface_near_peak_list)
        for j in range(len(x_surface_near_peak_list)):
            x_surface_near_peak=x_surface_near_peak_list[j]
            m_surface_near_peak=m_surface_near_peak_list[j]
            judge_tmp=0
            if mode_finder_obj.x_min<=x_surface_near_peak and x_surface_near_peak<=mode_finder_obj.x_max:
                index=np.argmin(abs(mode_finder_obj.x-x_surface_near_peak))
                
                omega_e_plasma_kHz=float(n)*mode_finder_obj.ome[index]
                Doppler_kHz=float(n)*mode_finder_obj.Doppler[index]
                omega_e_lab_kHz=omega_e_plasma_kHz+Doppler_kHz

                omega_e_plasma_kHz_list.append(omega_e_plasma_kHz)
                omega_e_lab_kHz_list.append(omega_e_lab_kHz)
                Doppler_kHz_list.append(Doppler_kHz)
                k_tmp=0
                for k in range(len(Frequency_list)):
                    f=Frequency_list[k]
                    f_error=abs(f-omega_e_lab_kHz)/f
                    #print('f_error')
                    #print(f_error)
                    if f_error<=q_scan_f_err:
                        judge_list[k]=1
                        k_tmp=k
                        mu_list_temp.append(weight_list[k]*abs(x_surface_near_peak-mode_finder_obj.x_peak))
                if reject_band_outside_for_q_scan:                
                    if not mode_finder_obj.\
                            inside_freq_band_check(\
                                omega_e_lab_kHz,\
                                freq_min_list,\
                                freq_max_list):
                        judge_list[k_tmp]=0


    if np.prod(judge_list)==1:
        #print('q_scale')
        #print(q_scale)
        q_scale_works_list.append(q_scale)
        q_shift_works_list.append(q_shift)
        shat_scale_works_list.append(shat_scale)
        mu_works_list.append(np.mean(mu_list_temp))
        if show_plot==True:
            mode_finder_obj.Plot_ome_q_surface_frequency_list(\
                peak_percent,n_min,n_max,Frequency_list,Frequency_error=q_scan_f_err,\
                save_imag=True,\
                image_name=Output_Path+f'q_scale={q_scale:.3f}_q_shift={q_shift:.3f}_shat_scale={shat_scale:.3f}.jpg')

print('********mu_works_list********')
print(mu_works_list)
print('********q_scale_works_list********')
print(q_scale_works_list)
if len(mu_works_list)==0:
    print('No possible variation of q profile for the given frequency range, please check the frequency or the frequency_error.')
    exit()

if scan_mode==0:
    index=np.argmin(mu_works_list)

    q_scale=q_scale_works_list[index]
    q_shift=q_shift_works_list[index]
    shat_scale=shat_scale_works_list[index]

    q_scale_works_list=[q_scale]
    q_shift_works_list=[q_shift]
    shat_scale_works_list=[shat_scale]

    mode_finder_obj.reset_profile()
    mode_finder_obj.modify_profile(q_scale=q_scale,q_shift=q_shift,\
                                    shat_scale=shat_scale,\
                                    ne_scale=1.,te_scale=1.,\
                                    ne_shift=0.,te_shift=0.,\
                                    Doppler_scale=1.,\
                                    show_plot=False)

    #mode_finder_obj.q_back_to_nominal()
    #mode_finder_obj.q_modify(q_scale,q_shift)
    if show_plot==True:
        mode_finder_obj.Plot_ome_q_surface_frequency_list(peak_percent,n_min,n_max,Frequency_list,\
                            Frequency_error=Frequency_error,save_imag=False)

elif scan_mode==1:
    pass 
elif scan_mode==2:
    q_scale_works_list=q_scale_list
    q_shift_works_list=q_shift_list
    shat_scale_works_list=shat_scale_list



#***step 2:

freq_min_list=[f*(1.-Frequency_error) for f in Frequency_list]
freq_max_list=[f*(1.+Frequency_error) for f in Frequency_list]



if Run_mode==6:
    from SLiM_NN.Dispersion_NN import Dispersion_NN
    Dispersion_NN_obj=Dispersion_NN(NN_path)


def calc_profile(q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift):
    mode_finder_obj.reset_profile()
    mode_finder_obj.modify_profile(q_scale=q_scale,q_shift=q_shift,\
                                    shat_scale=shat_scale,\
                                    ne_scale=ne_scale,te_scale=te_scale,\
                                    ne_shift=ne_shift,te_shift=te_shift,\
                                    Doppler_scale=1.,\
                                    show_plot=False)
    #mean_rho,xstar=mode_finder_obj.omega_gaussian_fit(manual=False)
    mean_rho,xstar=mode_finder_obj.omega_gaussian_fit(manual=False,fit_type=1)
    mode_finder_obj.set_xstar(xstar)
    #print(mode_finder_obj)

    #generating the parameter list
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
    shat_scale_list0=[]
    ne_scale_list0=[]
    ne_shift_list0=[]
    te_scale_list0=[]
    te_shift_list0=[]
    shat_scale_list0=[]

    Output_suffix=f'_q_scale={q_scale:.3f}_q_shift={q_shift:.3f}_shat_scale={shat_scale:.3f}_ne_scale={ne_scale:.3f}_ne_shift={ne_shift:.3f}_te_scale={te_scale:.3f}_te_shift={te_shift:.3f}'

    if Run_mode==1:#simple rational surface alignment
        ModIndex=1
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

    for n in n0_list:
        x_surface_near_peak_list, m_surface_near_peak_list=mode_finder_obj.Rational_surface_top_surfaces(n,top=surface_num)
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
                shat_scale_list0.append(shat_scale)
                ne_scale_list0.append(ne_scale)
                ne_shift_list0.append(ne_shift)
                te_scale_list0.append(te_scale)
                te_shift_list0.append(te_shift)
    
    
    d = {'q_scale':q_scale_list0,'q_shift':q_shift_list0,\
        'shat_scale':shat_scale_list0,\
        'ne_scale':ne_scale,'te_scale':te_scale,\
        'Doppler_shift_scale':[1]*len(n_list),\
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
    df=pd.DataFrame(d, columns=['q_scale','q_shift','shat_scale',\
        'ne_scale','te_scale','n','m','rho_tor',\
        'omega_plasma_kHz','omega_lab_kHz','gamma_cs_a','omega_n_kHz',\
        'omega_n_cs_a','omega_e_plasma_kHz','omega_e_lab_kHz',\
        'peak_percentage','nu','zeff','eta','shat','beta','ky',\
        'ModIndex','mu','xstar'])   #construct the panda dataframe
    
    judge_list=np.zeros(len(Frequency_list),dtype=int)

    file_name=Output_Path+filename

    if Run_mode==1:
        pass
    else:    
        with open(Output_Path+filename, 'w', newline='') as csvfile:     #clear all and then write a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow(['q_scale','q_shift',\
                'shat_scale',\
                'ne_scale','ne_shift',\
                'te_scale','te_shift',\
                'Doppler_shift_scale',\
                'n','m','rho_tor',\
                'omega_plasma_kHz','omega_lab_kHz',\
                'gamma_cs_a','omega_n_kHz',\
                'omega_n_cs_a','omega_e_plasma_kHz',\
                'omega_e_lab_kHz','peak_percentage',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
        csvfile.close()

    omega_lab_kHz_list=[]
    gamma_list=[]
    Doppler_list=[]
    for i in range(len(n_list)):
        #print(df['mu'][i])
        if Run_mode==1:
            w0=1+df['eta'][i]+10j
        elif Run_mode==4:
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

        #for testing
        #gamma_cs_a=1. 
        #omega_kHz=df['omega_e_plasma_kHz'][i]

        
        with open(Output_Path+filename, 'a+', newline='') as csvfile: #adding a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([ q_scale_list0[i],\
                q_shift_list0[i],\
                shat_scale_list0[i],\
                ne_scale_list0[i],\
                ne_shift_list0[i],\
                te_scale_list0[i],\
                te_shift_list0[i],\
                1.,\
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
        omega_lab_kHz_list.append(omega_kHz+df['omega_e_lab_kHz'][i]-df['omega_e_plasma_kHz'][i])
        Doppler_list.append(df['omega_e_lab_kHz'][i]-df['omega_e_plasma_kHz'][i])
        gamma_list.append(gamma_cs_a)
    
    for Doppler_shift_scale in Doppler_shift_scale_list:
        Frequency_error_list=[]
        omega_lab_kHz_work_list=[]
        best_f_list=[]
        best_f_error_list=[]
        best_f_error_reject_list=[]
        n_work_list=np.zeros(len(n_list_observe),dtype=int)

        #loop through the frequency observed in experiment
        for k in range(len(Frequency_list)):
            f_list=[]
            f_error_list=[]
            f_error_reject_list=[]
            f_tmp=0.
            f_error_tmp=99999999.
            #loop through the frequency calculated from the SLiM
            for j in range(len(omega_lab_kHz_list)):
                f=Frequency_list[k]
                f_slim=omega_lab_kHz_list[j]+Doppler_list[j]*(Doppler_shift_scale-1.)
                f_error=abs(f-f_slim)/f

                if f_error<=Frequency_error and gamma_list[j]>=0.0000001:
                    judge_list[k]=1
                    f_error_list.append(f_error)
                    f_list.append(f_slim)
                    if f_error_tmp>f_error:
                        f_tmp=f_slim
                        f_error_tmp=f_error

                if n_list[j]==n_list_observe[k]:
                    n_work_list[k]=1


            Frequency_error_list.append(f_error_list)
            omega_lab_kHz_work_list.append(f_list)
            best_f_list.append(f_tmp)
            best_f_error_list.append(f_error_tmp)

        best_f_error_avg=np.mean(np.array(best_f_error_list,dtype=float))

        df_scale=[
                q_scale_list0[i],
                q_shift_list0[i],
                shat_scale_list0[i],
                ne_scale_list0[i],
                ne_shift_list0[i],
                te_scale_list0[i],
                te_shift_list0[i],
                Doppler_shift_scale
                ]
        scale_score,total_score=total_score_calc(best_f_error_avg,n_work_list,df_scale)

        reject=0
        for j in range(len(omega_lab_kHz_list)):
            f=omega_lab_kHz_list[j]+Doppler_list[j]*(Doppler_shift_scale-1.)
            gamma=gamma_list[j]
            if (not mode_finder_obj.inside_freq_band_check(\
                f,freq_min_list,freq_max_list)) and gamma>=0.0000001:
                reject=1

        best_f_error_reject_avg=np.mean(np.array(best_f_error_reject_list,dtype=float))
    
        if np.prod(judge_list)!=1:
            Frequency_error_list='NA'

        with open(Output_Path+'0Equalibrium_summary.csv', 'a+', newline='') as csvfile:     #clear all and then write a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([q_scale,q_shift,\
                        shat_scale,\
                        ne_scale,ne_shift,\
                        te_scale,te_shift,\
                        Doppler_shift_scale,\
                        n_list,m_list,\
                        np.prod(judge_list),\
                        np.prod(n_work_list),\
                        omega_lab_kHz_work_list,\
                        Frequency_error_list,\
                        best_f_list,\
                        best_f_error_list,\
                        best_f_error_avg,\
                        reject,\
                        scale_score,\
                        total_score,\
                        file_name])
        csvfile.close()

        if (np.prod(judge_list)*np.prod(n_work_list)==1 and scan_mode==-1) or\
           (np.prod(judge_list)*np.prod(n_work_list)==1 and scan_mode==3):
            print('\n[q_scale,q_shift,\
                    shat_scale,\
                    ne_scale,ne_shift,\
                    te_scale,te_shift,\
                    Doppler_shift_scale]=\n'
                    +str([q_scale,q_shift,\
                        shat_scale,\
                        ne_scale,ne_shift,\
                        te_scale,te_shift,\
                        Doppler_shift_scale])
                                    )
            sys.exit()
            
        

print('*******Scaning*********')

if scan_mode==4:
    scale_list=[
    [1.,0.,shat_scale,ne_scale,ne_shift,te_scale,te_shift]\
        for shat_scale in shat_scale_works_list
        for ne_scale in ne_scale_list\
        for ne_shift in ne_shift_list\
        for te_scale in te_scale_list\
        for te_shift in te_shift_list
                    ]
    random.shuffle(scale_list)
    for [q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift] in tqdm(scale_list):
        mode_finder_obj.reset_profile()
        mode_finder_obj.modify_profile(q_scale=q_scale,q_shift=q_shift,\
                                    shat_scale=shat_scale,\
                                    ne_scale=ne_scale,te_scale=te_scale,\
                                    ne_shift=ne_shift,te_shift=te_shift,\
                                    Doppler_scale=1.,\
                                    show_plot=False)
        q_scale_align=mode_finder_obj.q_scale_for_alignment(q_at_peak)
        calc_profile(q_scale_align,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift)

elif len(q_scale_works_list)\
        *len(q_shift_works_list)\
        *len(shat_scale_works_list)\
        *len(ne_scale_list)\
        *len(ne_shift_list)\
        *len(te_scale_list)\
        *len(te_shift_list)>10000:
    if scan_mode==-1 or scan_mode==3:
        while(1==1):
            q_scale=random.choice(q_scale_list)
            q_shift=random.choice(q_shift_list)
            shat_scale=random.choice(shat_scale_list)
            ne_scale=random.choice(ne_scale_list)
            ne_shift=random.choice(ne_shift_list)
            te_scale=random.choice(te_scale_list)
            te_shift=random.choice(te_shift_list)
            print(str([q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift]))
            calc_profile(q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift)
    else:
        print('total run time too long, change the scan_mode')
        sys.exit()
else:
    scale_list=[
    [q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift]\
        for q_scale in q_scale_works_list\
        for q_shift in q_shift_works_list\
        for shat_scale in shat_scale_works_list
        for ne_scale in ne_scale_list\
        for ne_shift in ne_shift_list\
        for te_scale in te_scale_list\
        for te_shift in te_shift_list
                    ]
    random.shuffle(scale_list)

    for [q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift] in tqdm(scale_list):
        calc_profile(q_scale,q_shift,shat_scale,ne_scale,ne_shift,te_scale,te_shift)
