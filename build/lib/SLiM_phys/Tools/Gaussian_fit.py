import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#For GUI
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

#amplitude,mean,stddev=gaussian_fit(x,data,manual=False)
#amplitude,mean,stddev=gaussian_fit(x,data,manual=True)
#amplitude,mean,stddev=gaussian_fit_GUI(root,x,data)

def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / (1.*stddev))**2.)

def gaussian_fit(x,data,manual=False,fit_type=0):
    if manual==False:#auto_fit
        if fit_type==0:
            amplitude,mean,stddev=gaussian_fit_auto(x,data)
        elif fit_type==1:
            max_indx=np.argmax(data)
            mean=x[max_indx]
            amplitude=data[max_indx]
            amplitude_sigma=amplitude*np.exp(-1.)
            data_left=data[:max_indx]
            data_right=data[max_indx:]
            x_left=x[:max_indx]
            x_right=x[max_indx:]
            x_left_sigma=x_left[np.argmin(abs(data_left-amplitude_sigma))]
            x_right_sigma=x_right[np.argmin(abs(data_right-amplitude_sigma))]

            stddev=0.5*abs(x_left_sigma-x_right_sigma)
            

            if 1==0:
                print('amplitude,mean,stddev='+str([amplitude,mean,stddev]))
                amplitude_,mean_,stddev_=gaussian_fit_auto(x,data)
                print('amplitude,mean,stddev='+str([amplitude_,mean_,stddev_]))
                popt=[amplitude,mean,stddev]
                popt_=[amplitude_,mean_,stddev_]
                plt.clf()
                plt.plot(x,data)
                plt.axvline(x_left_sigma)
                plt.axvline(x_right_sigma)
                plt.plot(x,gaussian(x, *popt),color='red',label='fit_type 1')
                plt.plot(x,gaussian(x, *popt_),color='green',label='fit_type 0')
                plt.legend()
                plt.show()

    else: #manual_fit using GUI
        amplitude,mean,stddev=gaussian_fit_manual(x,data)
    return amplitude,mean,stddev

def gaussian_fit_GUI(root,x,data):
    x,data=np.array(x), np.array(data)

    global amplitude
    global mean
    global stddev

    top=tk.Toplevel(root)
    top.title('Gaussian Fit')
    #root.geometry("500x500")
    
    #load the icon for the GUI 
    try:
        top.iconbitmap('./SLiM.ico')
    except:
        pass

    #warnings.simplefilter("error", OptimizeWarning)

    #top=tk.LabelFrame(root, text='Gaussian Fit',padx=20,pady=20)
    #top.grid(row=0,column=1)

    frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',\
                            padx=20,pady=20)
    frame_plot.grid(row=0,column=0)
    fig = Figure(figsize = (5, 5),\
                dpi = 100)
    plot1 = fig.add_subplot(111)
    plot1.plot(x,data, label="data")

    try:
        popt, pcov = optimize.curve_fit(gaussian, x,data)  

        amplitude=popt[0]
        mean=popt[1]
        stddev=popt[2]

        plot1.plot(x, gaussian(x, *popt), label="fit")
        plot1.axvline(mean,color='red',alpha=0.5)
        plot1.axvline(mean+stddev,color='red',alpha=0.5)
        plot1.axvline(mean-stddev,color='red',alpha=0.5)
        plot1.legend()

    except RuntimeError:
        print("Curve fit failed, need to fit manually")
        
        max_index=np.argmax(data)

        amplitude=data[max_index]
        mean=x[max_index]

        data_left=data[:max_index]
        x_left=x[:max_index]
        x_std_index=np.argmin(abs( data_left-amplitude*np.exp(-1.) ))
        stddev=abs(mean-x_left[x_std_index])

        plot1.plot(x, gaussian(x, amplitude, mean, stddev), label="fit")
        plot1.axvline(mean,color='red',alpha=0.5)
        plot1.axvline(mean+stddev,color='red',alpha=0.5)
        plot1.axvline(mean-stddev,color='red',alpha=0.5)
        plot1.legend()

    canvas = FigureCanvasTkAgg(fig,master = frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,frame_plot)
    toolbar.update()
    canvas.get_tk_widget().pack()

    root1=tk.LabelFrame(top, text='User Box',padx=20,pady=20)
    root1.grid(row=0,column=1)


    frame1=tk.LabelFrame(root1, text='Accept/Reject the fit',padx=20,pady=20)
    frame1.grid(row=0,column=0)
    #import an option button
    opt_var1= tk.IntVar() #Integar Varible, Other options: StringVar()

    option_button11=tk.Radiobutton(frame1, text='Accept the fit',\
                                variable=opt_var1, value=1,\
                                command=lambda: click_button(opt_var1.get(), top,root1))
    option_button11.grid(row=1,column=0)

    option_button12=tk.Radiobutton(frame1, text='Enter manually',\
                                variable=opt_var1, value=2,\
                                command=lambda: click_button(opt_var1.get(), top,root1))
    option_button12.grid(row=2,column=0)
        

    
    def save_exit(amplitude,mean,stddev):
        looping=0
        top.quit()

    def click_button(value, top, root1):

        global amplitude
        global mean
        global stddev 

        label_list=[]

        frame_data=tk.LabelFrame(root1, text='Current fitting parameter',padx=80,pady=20)
        frame_data.grid(row=1,column=0)
        label_list.append(frame_data)

        amplitude_string=f'amplitude = {amplitude}'
        mean_string     =f'mean      = {mean}'
        std_string      =f'stddev    = {stddev}'
        amplitude_string=amplitude_string+' '*(50-len(amplitude_string))
        mean_string=mean_string+' '*(50-len(mean_string))
        std_string=std_string+' '*(50-len(std_string))

        amp_label =tk.Label(frame_data,text=amplitude_string)
        mean_label=tk.Label(frame_data,text=mean_string)
        std_label =tk.Label(frame_data,text=std_string)
        amp_label.grid(row=0,column=0)
        mean_label.grid(row=1,column=0)
        std_label.grid(row=2,column=0)
        label_list.append(amp_label)
        label_list.append(mean_label)
        label_list.append(std_label)

        def plot_manual_fit(x,data,mean_Input,sigma_Input,top,label_list):
            #frame_plot.grid_forget()

            frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',padx=20,pady=20)
            frame_plot.grid(row=0,column=0)
            fig = Figure(figsize = (5, 5),
                        dpi = 100)
            plot1 = fig.add_subplot(111)
            plot1.plot(x,data, label="data")
            
            global amplitude
            global mean
            global stddev 

            mean=float(mean_Input)
            amplitude=data[np.argmin(abs(x-mean))]
            stddev=float(sigma_Input)


            frame_data=label_list[0]
            amp_label=label_list[1]
            mean_label=label_list[2]
            std_label=label_list[3]

            amp_label.grid_forget()
            mean_label.grid_forget()
            std_label.grid_forget()

            amplitude_string=f'amplitude = {amplitude}'
            mean_string     =f'mean      = {mean}'
            std_string      =f'stddev    = {stddev}'

            amplitude_string=amplitude_string+' '*(50-len(amplitude_string))
            mean_string=mean_string+' '*(50-len(mean_string))
            std_string=std_string+' '*(50-len(std_string))


            amp_label =tk.Label(frame_data,text=amplitude_string)
            mean_label=tk.Label(frame_data,text=mean_string)
            std_label =tk.Label(frame_data,text=std_string)
            amp_label.grid(row=0,column=0)
            mean_label.grid(row=1,column=0)
            std_label.grid(row=2,column=0)

            plot1.plot(x, gaussian(x, amplitude, mean, stddev), label="fit")
            plot1.axvline(mean,color='red',alpha=0.5)
            plot1.axvline(mean+stddev,color='red',alpha=0.5)
            plot1.axvline(mean-stddev,color='red',alpha=0.5)
            plot1.legend()
        
            canvas = FigureCanvasTkAgg(fig,master = frame_plot)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas,frame_plot)
            toolbar.update()
            canvas.get_tk_widget().pack()

        

        if value==1:
            #'Accept the fit'
            myButton2=tk.Button(root1, text='Save and Continue', \
                command=lambda : save_exit(amplitude,mean,stddev))
            myButton2.grid(row=3,column=0)

        elif value==2:
            #Label1.grid_forget()
            #'Enter manually'
            frame_input=tk.LabelFrame(root1, text='Manual Input',padx=10,pady=10)
            frame_input.grid(row=2,column=0)
            tk.Label(frame_input,text='mu(center) = ').grid(row=0,column=0)
            mean_Input_box=tk.Entry(frame_input, width=30, bg='green', fg='white')
            mean_Input_box.insert(0,'')
            mean_Input_box.grid(row=0,column=1)
        
            tk.Label(frame_input,text='sigma(spread) = ').grid(row=1,column=0)
            sigma_Input_box=tk.Entry(frame_input, width=30, bg='green', fg='white')
            sigma_Input_box.insert(0,'')
            sigma_Input_box.grid(row=1,column=1)

            

            Plot_Button=tk.Button(frame_input, text='Plot the Manual Fit',\
                command=lambda: plot_manual_fit(x,data,\
                    float( mean_Input_box.get()  ),\
                    float( sigma_Input_box.get() ),\
                    top,label_list\
                    )  \
                )
            Plot_Button.grid(row=2,column=1)

            Save_Button=tk.Button(root1, text='Save and Continue',\
                     state=tk.DISABLED)#state: tk.DISABLED, or tk.NORMAL
            Save_Button.grid(row=3,column=0)

    try:
        stddev=abs(stddev)
    except:
        amplitude=0.
        mean=0.
        stddev=0.

    print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
    top.mainloop()

    
    return amplitude,mean,stddev

def gaussian_fit_auto(x,data):
    judge=0
    try:
        popt, pcov = optimize.curve_fit(gaussian, x,data)  
        #print(gaussian)
        #print(popt)
        #print(pcov)

        max_index=np.argmax(data)
        if 0==1:
            plt.clf()
            plt.plot(x,data, label="data")
            plt.plot(x, gaussian(x, *popt), label="fit")
            plt.axvline(x[max_index],color='red',alpha=0.5)
            plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
            plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
            plt.legend()
            plt.show()

        error_temp=np.sum(abs(data-gaussian(x, *popt)))/abs(np.sum(data))
        #print('norm_error='+str(error_temp))
        if error_temp<0.2:
            amplitude=popt[0]
            mean     =popt[1]
            stddev   =popt[2] 
        else:
            print("Curve fit failed, need to restrict the range")
            amplitude=0
            mean     =0
            stddev   =0
    except RuntimeError:
        print("Curve fit failed, need to restrict the range")
        max_index=np.argmax(data)

        amplitude=data[max_index]
        mean=x[max_index]

        data_left=data[:max_index]
        x_left=x[:max_index]
        x_std_index=np.argmin(abs( data_left-amplitude*np.exp(-1.) ))
        stddev=abs(mean-x_left[x_std_index])
        
    return amplitude,mean,stddev

def gaussian_fit_manual(x,data):
    x,data=np.array(x), np.array(data)

    #warnings.simplefilter("error", OptimizeWarning)
    judge=0

    try:
        popt, pcov = optimize.curve_fit(gaussian, x,data)  
        max_index=np.argmax(data)
        
        
        plt.clf()
        plt.plot(x,data, label="data")
        plt.plot(x, gaussian(x, *popt), label="fit")
        plt.axvline(x[max_index],color='red',alpha=0.5)
        plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
        plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
        plt.legend()
        plt.show()

        judge=int(input("Is the fit okay?  0. No 1. Yes"))

    except RuntimeError:
        print("Curve fit failed, need to restrict the range")

        judge=0
        
        popt=[0]*3  
        max_index=np.argmax(data)
        popt[0]=data[max_index] #amplitud
        popt[1]=x[max_index]    #mean

        plt.clf()
        plt.plot(x,data, label="data")
        plt.axvline(x[max_index],color='red',alpha=0.5)
        plt.legend()
        plt.show()

        while judge==0:

            popt[2]=float(input("sigma="))*np.sqrt(2.)  #stddev

            plt.clf()
            plt.plot(x,data, label="data")
            plt.plot(x, gaussian(x, *popt), label="fit")
            plt.axvline(x[max_index],color='red',alpha=0.5)
            plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
            plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
            plt.legend()
            plt.show()

            judge=int(input("Is the fit okay?  0. No 1. Yes"))


    while judge==0:

        popt=[0]*3  
        max_index=np.argmax(data)
        popt[0]=data[max_index] #amplitud
        popt[1]=x[max_index]    #mean

        popt[2]=float(input("sigma="))*np.sqrt(2.)  #stddev

        plt.clf()
        plt.plot(x,data, label="data")
        plt.plot(x, gaussian(x, *popt), label="fit")
        plt.axvline(x[max_index],color='red',alpha=0.5)
        plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
        plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
        plt.legend()
        plt.show()

        judge=int(input("Is the fit okay?  0. No 1. Yes"))

            
    amplitude=popt[0]
    mean     =popt[1]
    stddev   =popt[2] 

    return amplitude,mean,stddev

