import tkinter as tk #for GUI
from tkinter import messagebox  #for the finishing popup
from tkinter import filedialog  #for file and folder browsing 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk) #plot the Figure in GUI
from tqdm import tqdm  #for progress bar
import ast  #for converting the string representation of list to a list
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
Output_Path='./output/'

#global varible
global geomfile_name
geomfile_name=''
global profile_name
profile_name=''

q_uncertainty=0.1

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

#************start of n_tor********
tk.Label(Omega_frame,text='n_tor list').grid(row=2,column=0)
n_list_inputbox=tk.Entry(Omega_frame, width=20)
n_list_inputbox.insert(0,'[3,4,5]')
n_list_inputbox.grid(row=2,column=1)
#************end of n_tor range********

#************start of Unstable_list********
tk.Label(Omega_frame,text='Unstable_list').grid(row=3,column=0)
unstable_list_inputbox=tk.Entry(Omega_frame, width=20)
unstable_list_inputbox.insert(0,'[1,0,1]')
unstable_list_inputbox.grid(row=3,column=1)
#************end of color_list range********

#************start of color_list range********
tk.Label(Omega_frame,text='color_list').grid(row=4,column=0)
color_list_inputbox=tk.Entry(Omega_frame, width=20)
color_list_inputbox.insert(0,'-1')
color_list_inputbox.grid(row=4,column=1)
#************end of color_list range********

tk.Label(Omega_frame,text='q=').grid(row=5,column=0)
q_scale_inputbox=tk.Entry(Omega_frame, width=20)
q_scale_inputbox.insert(0,'1.')
q_scale_inputbox.grid(row=5,column=1)

tk.Label(Omega_frame,text='*q0+').grid(row=5,column=2)
q_shift_inputbox=tk.Entry(Omega_frame, width=20)
q_shift_inputbox.insert(0,'0.')
q_shift_inputbox.grid(row=5,column=3)

tk.Label(Omega_frame,text='m_i/m_p=').grid(row=6,column=0)
mref_inputbox=tk.Entry(Omega_frame, width=20)
mref_inputbox.insert(0,'2.')
mref_inputbox.grid(row=6,column=1)
#**********end omega percent*********************

def Load_data(profile_name,geomfile_name):
    profile_type=profile_type_var.get()
    geomfile_type=geomfile_type_var.get()

    omega_percent=float(omega_percent_inputbox.get())

    suffix=suffix_inputbox.get()

    global q_scale
    global q_shift

    q_scale=float(q_scale_inputbox.get())
    q_shift=float(q_shift_inputbox.get())

    
    print(f'q_scale,q_shift={q_scale},{q_shift}')

    x0_min=float(rho_tor_min_inputbox.get())
    x0_max=float(rho_tor_max_inputbox.get())

    mref=float(mref_inputbox.get())

    n_list=ast.literal_eval(n_list_inputbox.get())
    n_list=np.array(n_list,dtype=int)

    try:
        color_list=int(color_list_inputbox.get())
    except:
        color_list=ast.literal_eval(color_list_inputbox.get())
        color_list=np.array(color_list,dtype=str)

    unstable_list=ast.literal_eval(unstable_list_inputbox.get())
    unstable_list=np.array(unstable_list,dtype=int)
    

    print('omega_percent='+str(omega_percent)+'%')
    print('geomfile_name='+str(geomfile_name))
    print('profile_name ='+str(profile_name))
    print('geomfile_type='+str(geomfile_type))
    print('profile_type ='+str(profile_type))

    path=''

    try:
        outputpath=outputpath+'/'
    except: 
        outputpath='./Output/'

    #initidate the data
    mode_finder_obj=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max,mref=mref)

    top=tk.Toplevel(root)
    top.title('q modification')
    top.iconbitmap('./SLiM.ico')

    frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',\
                            padx=20,pady=20)
    frame_plot.grid(row=0,column=0)
    
    frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',\
                            padx=20,pady=20)
    frame_plot.grid(row=0,column=0)
    
    fig=mode_finder_obj.Plot_q_scale_rational_surfaces_colored_obj(omega_percent*0.01,\
                q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list)

    canvas = FigureCanvasTkAgg(fig,master = frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,frame_plot)
    toolbar.update()
    canvas.get_tk_widget().pack()

    root1=tk.LabelFrame(top, text='User Box',padx=20,pady=20)
    root1.grid(row=0,column=1)


    frame1=tk.LabelFrame(root1, text='Accept/Reject the modfication',padx=20,pady=20)
    frame1.grid(row=0,column=0)
    #import an option button
    opt_var1= tk.IntVar() #Integar Varible, Other options: StringVar()

    option_button11=tk.Radiobutton(frame1, text='Accept the modfication',\
                                variable=opt_var1, value=1,\
                                command=lambda: click_button(opt_var1.get(), top,root1))
    option_button11.grid(row=1,column=0)

    option_button12=tk.Radiobutton(frame1, text='Enter manually',\
                                variable=opt_var1, value=2,\
                                command=lambda: click_button(opt_var1.get(), top,root1))
    option_button12.grid(row=2,column=0)
        

    
    def save_exit(q_scale,q_shift):
        looping=0
        top.quit()

    def click_button(value, top, root1):

        global q_scale
        global q_shift 

        label_list=[]

        frame_data=tk.LabelFrame(root1, text='Current fitting parameter',padx=40,pady=20)
        frame_data.grid(row=1,column=0)
        label_list.append(frame_data)

        q_scale_string=f'q_scale = {q_scale}'
        q_shift_string=f'q_shift = {q_shift}'

        q_scale_string=q_scale_string+' '*(50-len(q_scale_string))
        q_shift_string=q_shift_string+' '*(50-len(q_shift_string))

        q_scale_label =tk.Label(frame_data,text=q_scale_string)
        q_shift_label=tk.Label(frame_data,text=q_shift_string)

        q_scale_label.grid(row=0,column=0)
        q_shift_label.grid(row=1,column=0)

        label_list.append(q_scale_label)
        label_list.append(q_shift_label)

        def plot_q_mod(mode_finder_obj,q_scale,q_shift,top,label_list):
            #frame_plot.grid_forget()

            frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',padx=20,pady=20)
            frame_plot.grid(row=0,column=0)

            fig=mode_finder_obj.Plot_q_scale_rational_surfaces_colored_obj(omega_percent*0.01,\
                q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list)

            frame_data=label_list[0]
            q_scale_label=label_list[1]
            q_shift_label=label_list[2]

            q_scale_label.grid_forget()
            q_shift_label.grid_forget()

            q_scale_string=f'q_scale = {q_scale}'
            q_shift_string=f'q_shift = {q_shift}'
            
            q_scale_string=q_scale_string+' '*(50-len(q_scale_string))
            q_shift_string=q_shift_string+' '*(50-len(q_shift_string))

            q_scale_label=tk.Label(frame_data,text=q_scale_string)
            q_shift_label=tk.Label(frame_data,text=q_shift_string)

            q_scale_label.grid(row=0,column=0)
            q_shift_label.grid(row=1,column=0)
        
            canvas = FigureCanvasTkAgg(fig,master = frame_plot)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas,frame_plot)
            toolbar.update()
            canvas.get_tk_widget().pack()

        

        if value==1:
            #'Accept the fit'
            myButton2=tk.Button(root1, text='Save and Continue', \
                command=lambda : save_exit(q_scale,q_shift))
            myButton2.grid(row=3,column=0)

        elif value==2:
            #Label1.grid_forget()
            #'Enter manually'
            frame_input=tk.LabelFrame(root1, text='Manual Input',padx=10,pady=10)
            frame_input.grid(row=2,column=0)
            tk.Label(frame_input,text='q = ').grid(row=0,column=0)
            q_scale_Input_box=tk.Entry(frame_input, width=10, bg='green', fg='white')
            q_scale_Input_box.insert(0,'')
            q_scale_Input_box.grid(row=0,column=1)
        
            tk.Label(frame_input,text='*q0+').grid(row=0,column=2)
            q_shift_Input_box=tk.Entry(frame_input, width=10, bg='green', fg='white')
            q_shift_Input_box.insert(0,'')
            q_shift_Input_box.grid(row=0,column=3)

            
            Plot_Button=tk.Button(frame_input, text='Plot the Manual Change',\
                command=lambda: plot_q_mod(mode_finder_obj,\
                    float(q_scale_Input_box.get()),\
                    float(q_shift_Input_box.get()),\
                    top,label_list\
                    )  \
                )
            Plot_Button.grid(row=1,column=1)

            Save_Button=tk.Button(root1, text='Save and Continue',\
                     state=tk.DISABLED)#state: tk.DISABLED, or tk.NORMAL
            Save_Button.grid(row=3,column=0)

    print(f'q_scale,q_shift={q_scale},{q_shift}')
    top.mainloop()




Print_Setting_Button=tk.Button(root, text='Run',\
                    command=lambda: Load_data(profile_name,\
                    geomfile_name),padx=50, pady=10)

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

#*******************Show all the setting and load the data********************

#creat the GUI

windows.mainloop()