# -*- coding: utf-8 -*-
"""
Created on 06/19/2021
Updated on 06/23/2021

@author: maxcurie
"""
import tkinter as tk
from tkinter import filedialog 
import numpy as np
import math

from MTMDispersion_tools import Parameter_reader
from MTMDispersion_tools import Peak_of_drive
from MTMDispersion_tools import Dispersion_n_scan
from MTMDispersion_tools import Spectrogram_2_frames


windows=tk.Tk()
windows.title('Mode finder')
windows.iconbitmap('./Physics_helper_logo.ico')

#global varible
global geomfile_name
geomfile_name=''
global profile_name
profile_name=''
global Run_mode
Run_mode=1


root=tk.LabelFrame(windows, text='Inputs',padx=5,pady=5)
root.grid(row=0,column=0)

#*************Input file*****************
Input_frame=tk.LabelFrame(root, text='Input files',padx=5,pady=5)
Input_frame.grid(row=0,column=0)

#**********Profile file setup****************************
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

#**********Profile file setup****************************



#**********Geometry file setup****************************
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

#**********Geometry file setup****************************
#*************Input file*****************


#*************Setting********************
Setting_frame=tk.LabelFrame(root, text='Setting',padx=5,pady=5)
Setting_frame.grid(row=1,column=0)

#**********omega percent*********************

Omega_frame=tk.LabelFrame(Setting_frame, text='Other Setting',padx=5,pady=5)
Omega_frame.grid(row=0,column=1)

tk.Label(Omega_frame,text='Omega* top percentage = ').grid(row=0,column=0)

omega_percent_inputbox=tk.Entry(Omega_frame, width=20)
omega_percent_inputbox.insert(0,'10.')
omega_percent_inputbox.grid(row=0,column=1)
tk.Label(Omega_frame,text='%').grid(row=0,column=2)



tk.Label(Omega_frame,text='Frequency Range').grid(row=1,column=0)
Freq_min_inputbox=tk.Entry(Omega_frame, width=20)
Freq_min_inputbox.insert(0,'0')
Freq_min_inputbox.grid(row=1,column=1)

tk.Label(Omega_frame,text='kHz~').grid(row=1,column=2)
Freq_max_inputbox=tk.Entry(Omega_frame, width=20)
Freq_max_inputbox.insert(0,'200')
Freq_max_inputbox.grid(row=1,column=3)

tk.Label(Omega_frame,text=' kHz').grid(row=1,column=4)

tk.Label(Omega_frame,text='Zeff(-1 for auto calculation)=').grid(row=2,column=0)
Zeff_inputbox=tk.Entry(Omega_frame, width=20)
Zeff_inputbox.insert(0,'2.33')
Zeff_inputbox.grid(row=2,column=1)

#**********omega percent*********************


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

#***********************run mode****************



#*************Setting********************



#*******************Show all the setting and load the data********************
omega_percent=float(omega_percent_inputbox.get())

q_scale=1.
manual_ped=-1
manual_zeff=-1
Z=6.
suffix='.dat'

def Load_data(profile_name,geomfile_name,Run_mode,\
    q_scale,manual_ped,manual_zeff):
    
    profile_type=profile_type_var.get()
    geomfile_type=geomfile_type_var.get()

    Freq_min=float(Freq_min_inputbox.get())
    Freq_max=float(Freq_max_inputbox.get())

    print('omega_percent='+str(omega_percent)+'%')
    print('Run_mode=     '+str(Run_mode))
    print('geomfile_name='+str(geomfile_name))
    print('profile_name ='+str(profile_name))
    print('geomfile_type='+str(geomfile_type))
    print('profile_type ='+str(profile_type))
    print('Freq_min=     '+str(Freq_min))
    print('Freq_max=     '+str(Freq_max))

    manual_zeff=float(Zeff_inputbox.get())
    uni_rhot,nu,eta,shat,beta,ky,q,mtmFreq,\
        omegaDoppler,omega_n,omega_n_GENE,\
        xstar,Lref, R_ref, rhoref, Zeff=\
            Parameter_reader(profile_type,profile_name,\
                geomfile_type,geomfile_name,Run_mode,\
                q_scale,manual_ped,manual_zeff,suffix,windows,Z=6.,\
                plot=False,output_csv=True)
    peak_index=np.argmax(mtmFreq)
    print(Freq_min)
    print(mtmFreq[peak_index]+omegaDoppler[peak_index])
    print(Freq_min/(mtmFreq[peak_index]+omegaDoppler[peak_index]))
    n_min=math.floor( Freq_min/(mtmFreq[peak_index]+omegaDoppler[peak_index]) )
    if n_min<=0:
        n_min=1
    n_max=math.ceil ( Freq_max/(mtmFreq[peak_index]+omegaDoppler[peak_index]) )
    x_peak_range, x_range_ind=Peak_of_drive(uni_rhot,mtmFreq,omegaDoppler,omega_percent)    
    x_list,n_list,m_list,gamma_list,\
        omega_list,gamma_list_kHz,\
        omega_list_kHz,omega_list_Lab_kHz,\
        omega_star_list_kHz,omega_star_list_Lab_kHz\
            =Dispersion_n_scan(uni_rhot,nu,eta,shat,beta,ky,q,\
                omega_n,omega_n_GENE,mtmFreq,omegaDoppler,\
                x_peak_range,x_range_ind,n_min,n_max,rhoref,\
                Lref,Run_mode,xstar,Zeff,plot=True,output_csv=True)
    
    bins=10

    f_lab,gamma_f_lab,f_plasma,gamma_f_plasma=\
        Spectrogram_2_frames(gamma_list_kHz,omega_star_list_kHz,omega_star_list_Lab_kHz,bins,plot=True)

    response = tk.messagebox.showinfo('All the output are in the current folder', 'Finished!')
    tk.Label(root,text=response).pack()




Print_Setting_Button=tk.Button(root, text='Run',\
                    command=lambda: Load_data(profile_name,geomfile_name,Run_mode,\
                                    q_scale,manual_ped,manual_zeff),\
                    padx=50, pady=10)

Print_Setting_Button.grid(row=3,column=0)

cite=tk.LabelFrame(root, text='Citation',padx=5,pady=5)
cite.grid(row=4,column=0)

tk.Label(cite,text='Please cite the following paper if you use this package: ',\
        font = "Helvetica 12 bold").grid(row=0,column=0)

import webbrowser
def callback(url):
    webbrowser.open_new(url)
link1 = tk.Label(cite, text="1. J. L. Larakers, M. Curie, D. R. Hatch, \
R. D. Hazeltine, and S. M. Mahajan(2021)\n\
Global Theory of Microtearing Modes in the \
Tokamak Pedestal    Physics Review Letter", fg="blue", cursor="hand2")
link1.grid(row=1,column=0)
link1.bind("<Button-1>", lambda e: callback("https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.225001"))
#*******************Show all the setting and load the data********************

#creat the GUI

windows.mainloop()

