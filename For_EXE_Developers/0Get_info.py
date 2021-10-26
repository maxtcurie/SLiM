import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import sys as sys

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************
q_scale_list=np.arange(1.02,1.04,0.0025)
q_shift_list=[0.]*len(q_scale_list)
n=6         #toroidal mode number of interest

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

manual_fit=False #Change to False for auto fit for xstar

mref=2.
profile_uncertainty=0.2
doppler_uncertainty=0.5
#************End of User Block*****************
#**********************************************

mode_finder_obj=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)
mode_finder_obj.ome_peak_range(peak_percent)
x,data=mode_finder_obj.x, mode_finder_obj.ome
mean_rho,xstar=mode_finder_obj.omega_gaussian_fit(manual=manual_fit)
mode_finder_obj.set_xstar(xstar)

with open(outputpath+'paramter_list_q_mod.csv', 'w', newline='') as csvfile:     #clear all and then write a row
    data = csv.writer(csvfile, delimiter=',')
    data.writerow(['n','m','x','nu','zeff','eta','shat','beta','ky','mu','xstar'])
csvfile.close()

for i in range(len(q_scale_list)):
    q_scale=q_scale_list[i]
    q_shift=q_shift_list[i]
    mode_finder_obj.q_back_to_nominal()
    mode_finder_obj.q_modify(q_scale,q_shift)
    x_surface_near_peak, m_surface_near_peak=\
        mode_finder_obj.Rational_surface_peak_surface(n)
    #mu negative for the rational surfaces on the left of the peak,\
    #   positive for the ones on the right
    nu,zeff,eta,shat,beta,ky,mu,xstar=\
        mode_finder_obj.parameter_for_dispersion(x_surface_near_peak)

    with open(outputpath+'paramter_list_q_mod.csv', 'a+', newline='') as csvfile:     #clear all and then write a row
        data = csv.writer(csvfile, delimiter=',')
        data.writerow([n,m_surface_near_peak,x_surface_near_peak,\
                        nu,zeff,eta,shat,beta,ky,mu,xstar])
    csvfile.close()
