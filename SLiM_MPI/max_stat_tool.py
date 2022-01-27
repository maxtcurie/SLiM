import optparse as op
import math
import cmath
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#********************************************************
#***********Define the function**************************
#Function average:
#Use the weighted average to "smooth" out the bump and rescaling
def smooth(avg_list,bin_size):
    l_list=len(avg_list)
    list_avg=np.zeros(l_list-bin_size+1)
    list_dev=np.zeros(l_list-bin_size+1)
    avg=0
    for i in range(0,len(list_avg)):
      list_avg[i]=np.mean(avg_list[i:i+bin_size]) 
      list_dev[i]=np.std(avg_list[i:i+bin_size]) 

    return list_avg, list_dev


def avg(avg_list,bin_size):
#weighted average check chapter 7 in an introduction to error analysisby John R. Taylor
  l_list=len(avg_list)
  #print avg_list
  #print 'l_list', l_list
  avg=0
  dev_sum=0
  #print 'Check point', l_list-bin_size+1
  for i in range(0,l_list-bin_size+1):
    #print i
    avg_temp=np.mean(avg_list[i:i+bin_size])
    std_temp=np.std(avg_list[i:i+bin_size])
    #print avg_temp
    #print std_temp
    avg=avg+avg_temp/(std_temp)**2
    dev_sum=dev_sum+1/(std_temp)**2
  if dev_sum == 0: 
    avg=avg
    dev_sum=10**20
  else:
    avg=avg/dev_sum #average
    dev_sum=np.sqrt(1/dev_sum)
  
  output=np.zeros(2)
  output[0]=avg
  output[1]=dev_sum
  return(output)

def avg_dev(avg_list,dev_list):
  bin_size=3 #bin size
  l_list=len(avg_list)
  avg=0
  dev=0
  dev_sum=0
  for i in range(0,l_list-bin_size+1):
    avg_temp=np.mean(avg_list[i:i+bin_size])
    dev_temp=np.mean(dev_list[i:i+bin_size])
    std_temp=np.std(avg_list[i:i+bin_size])
    avg=avg+avg_temp/(std_temp)**2
    dev=dev+dev_temp/(std_temp)**2
    dev_sum=dev_sum+1/(std_temp)**2
  avg=avg/dev_sum #average
  dev=dev/dev_sum #standard dev
  output=np.zeros(2)
  output[0]=avg
  output[1]=dev
  return(output)

def norm(a_list): #Normalized to the 1
  return(a_list/np.max(abs(a_list)))


#def step(list,center,): 

def zoom1D(x,y,x_min,x_max):
#this function zoon in the a 1D plot
  x_zoom=[]
  y_zoom=[]
  for i in range(len(x)):
    if x[i]<=x_max and x[i]>=x_min:
      x_zoom.append(x[i])
      y_zoom.append(y[i])
  return x_zoom, y_zoom

def zoom2D(x,y,z,x_min,x_max,y_min,y_max):
#this function zoon in the a 1D plot
  x_zoom=[]
  y_zoom=[]
  z_zoom=[]
  for i in range(len(x)):
    if x[i]<=x_max and x[i]>=x_min:
      if y[i]<=y_max and y[i]>=y_min:
        x_zoom.append(x[i])
        y_zoom.append(y[i])
        z_zoom.append(z[i])
  return x_zoom, y_zoom, z_zoom

#this function is from unutbu https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest_index(array, value): #this function return the index of the a value(or nearest)
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

def find_nearest(array, value): #this function return the index of the a value(or nearest)
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return array[idx]


def loop(x,f,x_min,x_max):#this function make the function return [sum f(x+n*x0)] where x0 is the period
  x0=x_max-x_min

  x = np.asarray(x)#same as find_nearest_index
  nx_min = (np.abs(x - x_min)).argmin()#same as find_nearest_index
  nx_max = (np.abs(x - x_max)).argmin()#same as find_nearest_index

  x_loop=x[nx_min:nx_max]
  f_loop=np.zeros(len(x_loop))

  x_loop = np.asarray(x_loop)
  for i in range(len(x)):
    xtemp=(x[i]-x_min)%x0+x_min
    nxtemp = np.argmin(np.abs(x_loop - xtemp)) #same as find_nearest_index
    f_loop[nxtemp]=f_loop[nxtemp]+f[i]

  return x_loop, f_loop


#x_sort,f_sort = sort_x_f(x,f)
def sort_x_f(x_unsort,f_unsort): 
   
    arr_unsort=[x_unsort,f_unsort]
    f_x_unsort=tuple(map(tuple, np.transpose(arr_unsort)))
      
    f_x_sort=sorted(f_x_unsort, key=lambda f_x_unsort: f_x_unsort[0])
    f_x_sort=np.array(f_x_sort)
    f_x_sort=np.transpose(f_x_sort)
    x_sort=f_x_sort[0,:]
    f_sort=f_x_sort[1,:]

    return x_sort,f_sort


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5*((x - mean) /stddev)**2)

def gaussian_fit(x,data):
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

            popt[2]=float(input("sigma="))  #stddev

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

        popt[2]=float(input("sigma="))  #stddev

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
    mean     =abs(popt[1])
    stddev   =abs(popt[2])

    return amplitude,mean,stddev

# p=np.polyfit(uni_rhot,q,5)

def Poly_fit(x, data, order, show):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """

    #https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    coeffs=np.polyfit(x,data,order)
    coeffs=np.flip(coeffs)

    o = len(coeffs)
    print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i

    if show==True:
        plt.clf()
        plt.plot(x,y,label='fit')
        plt.plot(x,data,label='original')
        plt.legend()
        plt.show()

    return coeffs
def coeff_to_Poly(x, x0, coeff, show):

    y = 0
    for i in range(len(coeff)):
        y += coeff[i]*(x-x0)**i

    if show==True:
        plt.clf()
        plt.plot(x,y)
        plt.show()

    return y