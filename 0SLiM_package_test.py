
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys as sys

from SLiM_phys.SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************
n_list       =[3,4,5]   #list of toroidal mode numbers
unstable_list=[1,0,1]   #list of wheather the mode number has the unstable MTM
color_list   =['orange','green','red']#list of color for the mode number
q_scale=0.98
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
peak_percent=0.01
order=10            # order of polynomial for q profile fit 

marker_size=5
font_size=15
q_uncertainty=0.1
mref=2.
profile_uncertainty=0.2
doppler_uncertainty=0.5
#************End of User Block*****************
#**********************************************

a=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)

a.Plot_q_scale_rational_surfaces_colored(peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list)
