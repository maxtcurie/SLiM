# -*- coding: utf-8 -*-
"""
Created on 06/19/2021
Updated on 06/23/2021

@author: maxcurie
"""
import tkinter as tk #for GUI
from tkinter import messagebox  #for the finishing popup
from tkinter import filedialog  #for file and folder browsing 
from tqdm import tqdm  #for progress bar
import numpy as np
import pandas as pd
import csv
import math

from SLiM_obj import mode_finder


windows=tk.Tk()
windows.title('Mode finder')
try:
    windows.iconbitmap('./SLiM.ico')
except:
    pass

#global varible
global geomfile_name
geomfile_name=''
global profile_name
profile_name=''
global Run_mode
Run_mode=1


root=tk.LabelFrame(windows, text='Inputs',padx=5,pady=5)

root.grid(row=0,column=0)

#*************start of Input file*****************
Input_frame=tk.LabelFrame(root, text='Input files',padx=5,pady=5)
Input_frame.grid(row=0,column=0)

#**********start of Profile file setup****************************
p_frame=tk.LabelFrame(Input_frame, text='Profile file',padx=5,pady=5)
p_frame.grid(row=0,column=0)
tk.Label(p_frame,text='Profile file type').grid(row=0,column=0)

#add the dropdown menu
profile_type_var=tk.StringVar()
profile_type_var.set('Choose type')
p_Options=['ITERDB','pfile','profile_e','profile_both']
p_drop=tk.OptionMenu(p_frame, profile_type_var, *p_Options)
p_drop.grid(row=0,column=1)


p_Path=tk.Label(p_frame,text='Profile file path')
p_Path.grid(row=1,column=0)

def p_Click():
    #find the file
    global p_frame
    global profile_name
    profile_name=filedialog.askopenfilename(initialdir='./',\
                                        title='select a file', \
                                        filetypes=( 
                                            ('all files', '*'),\
                                            ('ITERDB','.ITERDB' or '.iterdb'),\
                                            ('pfile','p*'),\
                                            ('profile','profile*'),\
                                            ) \
                                        )
    global profile_name_box
    profile_name_box=tk.Entry(p_frame, width=100)
    max_len=100
    if len(profile_name)>max_len:
        profile_name_box.insert(0,'...'+profile_name[-max_len:])
    else:
        profile_name_box.insert(0,profile_name)
    profile_name_box.grid(row=2,column=0)
    

#creat button
p_Path_Button=tk.Button(p_frame, text='Browse for Profile file path',\
                    command=p_Click,\
                    padx=50, pady=10)

p_Path_Button.grid(row=3,column=0)

#**********end of Profile file setup****************************

#**********start of Geometry file setup****************************
g_frame=tk.LabelFrame(Input_frame, text='Geometry file',padx=5,pady=5)
g_frame.grid(row=1,column=0)
tk.Label(g_frame,text='Geometry file type').grid(row=0,column=0)

#add the dropdown menu
geomfile_type_var=tk.StringVar()
geomfile_type_var.set('Choose type')
g_Options=['gfile','GENE_tracer']
g_drop=tk.OptionMenu(g_frame, geomfile_type_var, *g_Options)
g_drop.grid(row=0,column=1)


#find the file
g_Path=tk.Label(g_frame,text='Geometry file path')
g_Path.grid(row=1,column=0)

def g_Click():
    #find the file
    global g_frame
    global geomfile_name
    geomfile_name=filedialog.askopenfilename(initialdir='./',\
                                    title='select a file', \
                                    filetypes=( 
                                        ('all files', '*'),
                                        ('gfile/efit','g*')\
                                        ) \
                                    )
    global geomfile_name_box
    geomfile_name_box=tk.Entry(g_frame,width=100)

    
    if len(geomfile_name)>30:
        geomfile_name_box.insert(0,'...'+geomfile_name[-100:])
    else:
        geomfile_name_box.insert(0,geomfile_name)
    
    geomfile_name_box.grid(row=2,column=0)

#creat button
g_Path_Button=tk.Button(g_frame, text='Browse for Geometry file path',\
                    command=g_Click,\
                    padx=50, pady=10)

g_Path_Button.grid(row=3,column=0)


suffix_frame=tk.LabelFrame(g_frame, text='For GENE_tracer',padx=5,pady=5)
suffix_frame.grid(row=4,column=0)

tk.Label(suffix_frame,text='suffix(for GENE_tracer)=').grid(row=0,column=0)
suffix_inputbox=tk.Entry(suffix_frame, width=20)
suffix_inputbox.insert(0,'.dat')
suffix_inputbox.grid(row=0,column=1)
#**********end of Geometry file setup****************************
#*************end of Input file*****************


#*************start of Output file*****************
Output_frame=tk.LabelFrame(root, text='output files',padx=5,pady=5)
Output_frame.grid(row=1,column=0)

#*************start of output path*************
csv_frame=tk.LabelFrame(Output_frame, text='csv file',padx=5,pady=5)
csv_frame.grid(row=1,column=0)


#find the file
csv_Path=tk.Label(csv_frame,text='csv file path')
csv_Path.grid(row=1,column=0)

outputpath_box=tk.Entry(csv_frame, width=100)
max_len=100
outputpath_box.insert(0,'./Output')
outputpath_box.grid(row=2,column=0)

def csv_Click():
    #find the file
    global csv_frame
    global outputpath
    outputpath=filedialog.askdirectory(initialdir='./',\
                                    title='select a folder')

    outputpath_box=tk.Entry(csv_frame, width=100)
    max_len=100

    if len(outputpath)>max_len:
        outputpath_box.insert(0,'...'+outputpath[-1*max_len:])
    else:
        outputpath_box.insert(0,outputpath)
    
    outputpath_box.grid(row=2,column=0)

#creat button
csv_Button=tk.Button(csv_frame, text='Browse for csv file output path',\
                    command=csv_Click,\
                    padx=50, pady=10)

csv_Button.grid(row=3,column=0)

#*************end of output path*************

#*************end of Output file*****************

#*************Setting********************
Setting_frame=tk.LabelFrame(root, text='Setting',padx=5,pady=5)
Setting_frame.grid(row=2,column=0)

#**********omega percent*********************

Omega_frame=tk.LabelFrame(Setting_frame, text='Other Setting',padx=5,pady=5)
Omega_frame.grid(row=0,column=1)

tk.Label(Omega_frame,text='Omega* top percentage = ').grid(row=0,column=0)

omega_percent_inputbox=tk.Entry(Omega_frame, width=20)
omega_percent_inputbox.insert(0,'4.')
omega_percent_inputbox.grid(row=0,column=1)
tk.Label(Omega_frame,text='%').grid(row=0,column=2)

#************start of rho_tor range********
tk.Label(Omega_frame,text='rho_tor Range').grid(row=1,column=0)
rho_tor_min_inputbox=tk.Entry(Omega_frame, width=20)
rho_tor_min_inputbox.insert(0,'0.93')
rho_tor_min_inputbox.grid(row=1,column=1)

tk.Label(Omega_frame,text='~').grid(row=1,column=2)
rho_tor_max_inputbox=tk.Entry(Omega_frame, width=20)
rho_tor_max_inputbox.insert(0,'0.99')
rho_tor_max_inputbox.grid(row=1,column=3)
#************end of rho_tor range********

#************start of frequency range********
tk.Label(Omega_frame,text='Frequency Range').grid(row=2,column=0)
Freq_min_inputbox=tk.Entry(Omega_frame, width=20)
Freq_min_inputbox.insert(0,'0')
Freq_min_inputbox.grid(row=2,column=1)

tk.Label(Omega_frame,text='kHz~').grid(row=2,column=2)
Freq_max_inputbox=tk.Entry(Omega_frame, width=20)
Freq_max_inputbox.insert(0,'150')
Freq_max_inputbox.grid(row=2,column=3)

tk.Label(Omega_frame,text=' kHz').grid(row=2,column=4)
#************end of frequency range********

tk.Label(Omega_frame,text='q=').grid(row=3,column=0)
q_scale_inputbox=tk.Entry(Omega_frame, width=20)
q_scale_inputbox.insert(0,'1.')
q_scale_inputbox.grid(row=3,column=1)

tk.Label(Omega_frame,text='*q0+').grid(row=3,column=2)
q_shift_inputbox=tk.Entry(Omega_frame, width=20)
q_shift_inputbox.insert(0,'0.')
q_shift_inputbox.grid(row=3,column=3)

tk.Label(Omega_frame,text='Zeff(-1 for auto calculation)=').grid(row=4,column=0)
Zeff_inputbox=tk.Entry(Omega_frame, width=20)
Zeff_inputbox.insert(0,'-1')
Zeff_inputbox.grid(row=4,column=1)

tk.Label(Omega_frame,text='Impurity_charge=').grid(row=5,column=0)
Impurity_charge_inputbox=tk.Entry(Omega_frame, width=20)
Impurity_charge_inputbox.insert(0,'6.')
Impurity_charge_inputbox.grid(row=5,column=1)

tk.Label(Omega_frame,text='m_i/m_p=').grid(row=6,column=0)
mref_inputbox=tk.Entry(Omega_frame, width=20)
mref_inputbox.insert(0,'2.')
mref_inputbox.grid(row=6,column=1)

#**********end omega percent*********************


#***********************run mode****************

opt_var1= tk.IntVar() #Integar Varible, Other options: StringVar()
opt_var1.set(1)       #Set the default option as option1

def click_mode(a):
    global Run_mode
    Run_mode=a


frame1=tk.LabelFrame(Setting_frame, text='Running Mode Selection',padx=50,pady=40)
frame1.grid(row=0,column=0)

option_button11=tk.Radiobutton(frame1, text='Rational surface alignment(Fast)',\
                            variable=opt_var1, value=1,\
                            command=lambda: click_mode(opt_var1.get()))
option_button11.grid(row=1,column=0)

option_button12=tk.Radiobutton(frame1, text='Global Disperion Calculation',\
                            variable=opt_var1, value=2,\
                            command=lambda: click_mode(opt_var1.get()))
option_button12.grid(row=2,column=0)

option_button13=tk.Radiobutton(frame1, text='Local Disperion Calculation',\
                            variable=opt_var1, value=3,\
                            command=lambda: click_mode(opt_var1.get()))
option_button13.grid(row=3,column=0)

tk.Label(frame1,text='max number of rational surfaces for each n=').grid(row=4,column=0)
surface_num_inputbox=tk.Entry(frame1, width=20)
surface_num_inputbox.insert(0,'1')
surface_num_inputbox.grid(row=5,column=0)

#***********************run mode****************



#*************Setting********************



#*******************Show all the setting and load the data********************

def Load_data(profile_name,geomfile_name,Run_mode):
    profile_type=profile_type_var.get()
    geomfile_type=geomfile_type_var.get()
    omega_percent=float(omega_percent_inputbox.get())
    surface_num=int(surface_num_inputbox.get())

    global outputpath

    suffix=suffix_inputbox.get()

    q_scale=float(q_scale_inputbox.get())
    q_shift=float(q_shift_inputbox.get())

    x0_min=float(rho_tor_min_inputbox.get())
    x0_max=float(rho_tor_max_inputbox.get())
    
    Freq_min=float(Freq_min_inputbox.get())
    Freq_max=float(Freq_max_inputbox.get())

    mref=float(mref_inputbox.get())
    Impurity_charge=float(Impurity_charge_inputbox.get())
    manual_zeff=float(Zeff_inputbox.get())

    path=''
    try:
        outputpath=outputpath+'/'
    except: 
        outputpath='./Output/'

    print('omega_percent='+str(omega_percent)+'%')
    print('Run_mode=     '+str(Run_mode))
    print('geomfile_name='+str(geomfile_name))
    print('profile_name ='+str(profile_name))
    print('geomfile_type='+str(geomfile_type))
    print('profile_type ='+str(profile_type))
    print('Freq_min=     '+str(Freq_min))
    print('Freq_max=     '+str(Freq_max))
    print('outputpath=   '+str(outputpath))

    #initidate the data
    mode_finder_obj=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max,\
            manual_zeff,suffix,\
            mref,Impurity_charge\
            )
    mode_finder_obj.q_back_to_nominal()
    mode_finder_obj.q_modify(q_scale,q_shift)
    mode_finder_obj.ome_peak_range(omega_percent)


    peak_index=np.argmin(abs(mode_finder_obj.x-mode_finder_obj.x_peak))
    omega_e_peak_kHz=mode_finder_obj.ome[peak_index]

    omega_e_lab_peak_kHz=(mode_finder_obj.ome[peak_index]\
                +mode_finder_obj.Doppler[peak_index])
    n_min=math.floor( Freq_min/omega_e_lab_peak_kHz )
    if n_min<=0:
        n_min=1
    n_max=math.ceil( Freq_max/omega_e_lab_peak_kHz )

    mode_finder_obj.Plot_ome_q_surface_demo(omega_percent*0.01,\
        n_min,n_max,Freq_min,Freq_max)

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
    
    if Run_mode==1:#simple rational surface alignment
        ModIndex=-1
        filename='rational_surface_alignment.csv'
    if Run_mode==2:#global dispersion
        ModIndex=1
        filename='global_dispersion.csv'
        mean_rho,xstar=mode_finder_obj.omega_gaussian_fit_GUI(root,\
                    mode_finder_obj.x,mode_finder_obj.ome,\
                    mode_finder_obj.rho_s,\
                    mode_finder_obj.Lref)
        mode_finder_obj.set_xstar(xstar)
    elif Run_mode==3:#local dispersion
        ModIndex=0
        filename='local_dispersion.csv'
        mean_rho,xstar=mode_finder_obj.omega_gaussian_fit_GUI(root,\
                    mode_finder_obj.x,mode_finder_obj.ome,\
                    mode_finder_obj.rho_s,\
                    mode_finder_obj.Lref)
        mode_finder_obj.set_xstar(xstar)

    
    
    cs_to_kHz=mode_finder_obj.cs_to_kHz[peak_index]
    print('Finding the rational surfaces')
    n0_list=np.arange(n_min,n_max+1,1)
    for n in tqdm(n0_list):
        x_surface_near_peak_list, m_surface_near_peak_list=mode_finder_obj.Rational_surface_top_surfaces(n,top=surface_num)
        print(x_surface_near_peak_list)
        print(m_surface_near_peak_list)
        for i in range(len(x_surface_near_peak_list)):
            x_surface_near_peak=x_surface_near_peak_list[i]
            m_surface_near_peak=m_surface_near_peak_list[i]
            if mode_finder_obj.x_min<=x_surface_near_peak and x_surface_near_peak<=mode_finder_obj.x_max:
                nu,zeff,eta,shat,beta,ky,mu,xstar=\
                    mode_finder_obj.parameter_for_dispersion(x_surface_near_peak,n)
            
                index=np.argmin(abs(mode_finder_obj.x-x_surface_near_peak))
                omega_n_kHz=float(n)*mode_finder_obj.omn[index]
                omega_n_cs_a=float(n)*mode_finder_obj.omn[index]/cs_to_kHz
                omega_e_plasma_kHz=float(n)*mode_finder_obj.ome[index]
                omega_e_lab_kHz=float(n)*mode_finder_obj.ome[index]\
                            +float(n)*mode_finder_obj.Doppler[index]
            
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
    
    
    d = {'n':n_list,'m':m_list,'rho_tor':x_list,\
        'omega_plasma_kHz':[0]*len(n_list),\
        'omega_lab_kHz':[0]*len(n_list),\
        'gamma_cs_a':[0]*len(n_list),\
        'omega_n_kHz':omega_n_kHz_list,\
        'omega_n_cs_a':omega_n_cs_a_list,\
        'omega_e_plasma_kHz':omega_e_plasma_list,\
        'omega_e_lab_kHz':omega_e_lab_list,\
        'peak_percentage':omega_e_plasma_list/omega_e_peak_kHz,\
        'nu':nu_list,'zeff':[zeff]*len(n_list),'eta':eta_list,\
        'shat':shat_list,'beta':beta_list,'ky':ky_list,\
        'ModIndex':ModIndex_list,'mu':mu_list,'xstar':xstar_list}
    df=pd.DataFrame(d, columns=['n','m','rho_tor',\
        'omega_plasma_kHz','omega_lab_kHz','gamma_cs_a','omega_n_kHz',\
        'omega_n_cs_a','omega_e_plasma_kHz','omega_e_lab_kHz',\
        'peak_percentage','nu','zeff','eta','shat','beta','ky',\
        'ModIndex','mu','xstar'])   #construct the panda dataframe
    df.to_csv(outputpath+'parameter_list.csv',index=False)
        
    if Run_mode==1:
        pass
    else:
        with open(outputpath+filename, 'w', newline='') as csvfile:     #clear all and then write a row
            data = csv.writer(csvfile, delimiter=',')
            data.writerow(['n','m','rho_tor',\
                'omega_plasma_kHz','omega_lab_kHz',\
                'gamma_cs_a','omega_n_kHz',\
                'omega_n_cs_a','omega_e_plasma_kHz',\
                'omega_e_lab_kHz','peak_percentage',\
                'nu','zeff','eta','shat','beta','ky',\
                'ModIndex','mu','xstar'])
        csvfile.close()
    
        print('Calculate the dispersion relations')
        
        for i in tqdm(range(len(n_list))):
            w0=mode_finder_obj.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                    df['shat'][i],df['beta'][i],df['ky'][i],\
                    df['ModIndex'][i],df['mu'][i],df['xstar'][i])
            
            omega=np.real(w0)
            omega_kHz=omega*omega_n_kHz_list[i]
            gamma=np.imag(w0)
            gamma_cs_a=gamma*omega_n_cs_a_list[i]
            with open(outputpath+filename, 'a+', newline='') as csvfile: #adding a row
                data = csv.writer(csvfile, delimiter=',')
                data.writerow([ df['n'][i],df['m'][i],df['rho_tor'][i],\
                    omega_kHz,omega_kHz+df['omega_e_lab_kHz'][i]-df['omega_e_plasma_kHz'][i],gamma_cs_a,\
                    df['omega_n_kHz'][i],df['omega_n_cs_a'][i],\
                    df['omega_e_plasma_kHz'][i],df['omega_e_lab_kHz'][i],
                    df['peak_percentage'][i],df['nu'][i],\
                    df['zeff'][i],df['eta'][i],\
                    df['shat'][i],df['beta'][i],df['ky'][i],\
                    df['ModIndex'][i],df['mu'][i],df['xstar'][i] ])
            csvfile.close()
    
        
        
    
    response = tk.messagebox.showinfo('Finished!','All the output are in the output folder')
    tk.Label(root,text=response).pack()




Print_Setting_Button=tk.Button(root, text='Run',\
                    command=lambda: Load_data(profile_name,\
                    geomfile_name,Run_mode),padx=50, pady=10)

Print_Setting_Button.grid(row=4,column=0)

cite=tk.LabelFrame(root, text='Citation',padx=5,pady=5)
cite.grid(row=5,column=0)

tk.Label(cite,text='Please cite the following paper if you use this package: ',\
        font = "Helvetica 12 bold").grid(row=0,column=0)

import webbrowser
def callback(url):
    webbrowser.open_new(url)

link0 = tk.Label(cite, text="1. M. Curie, J. L. Larakers,  \
D. R. Hatch, O. Nelson, et al. (2021)\n\
Reduced predictive models for Micro-tearing modes \
in the pedestal APS DPP", fg="blue", cursor="hand2")
link0.grid(row=1,column=0)
link0.bind("<Button-1>", lambda e: callback("https://meetings.aps.org/Meeting/DPP21/Session/UI01.2"))


link1 = tk.Label(cite, text="2. J. L. Larakers, M. Curie, D. R. Hatch, \
R. D. Hazeltine, and S. M. Mahajan(2021)\n\
Global Theory of Microtearing Modes in the \
Tokamak Pedestal    Physics Review Letter", fg="blue", cursor="hand2")
link1.grid(row=2,column=0)
link1.bind("<Button-2>", lambda e: callback("https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.225001"))
#*******************Show all the setting and load the data********************

#creat the GUI

windows.mainloop()

