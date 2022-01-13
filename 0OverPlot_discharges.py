import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************

profile_name_list =['DIIID169510.01966_108.iterdb',\
                    'DIIID169510.03090_595.iterdb',\
                    '']
geomfile_name_list=['g169510.01966_108',\
                    'g169510.03090_595',
                    ]
path_list =['./Discharge_survey/D3D/169510/1966/',\
            './Discharge_survey/D3D/169510/3069/',\
            './Discharge_survey/D3D/169510/4069/']
profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"

outputpath='./Test_files/'
path='./Test_files/'


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

color_list=['orange','green','red']#list of color for the mode number
#************End of User Block*****************
#**********************************************

for i in range(len(profile_name_list)):
    profile_name=path+'p174819.03560'
    geomfile_name=path+'g174819.03560'
    a=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)

    a.Plot_q_scale_rational_surfaces_colored(peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list)
