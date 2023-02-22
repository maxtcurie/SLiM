import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys as sys
sys.path.insert(1, './../SLiM')
sys.path.insert(1, './../SLiM/Tools')

from SLiM_obj import mode_finder 

#**********************************************
#**********Start of User block*****************

profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
outputpath='./Test_files/'
path='./Test_files/'
profile_name=path+'p174819.03560'
geomfile_name=path+'g174819.03560'


x0_min=0.93         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor

#************End of User Block*****************
#**********************************************

a=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)
