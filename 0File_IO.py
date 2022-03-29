import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys as sys

from file_IO_obj import file_IO_obj

#**********************************************
#**********Start of User block*****************
profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
outputpath='./Test_files/'
inputpath='./Test_files/'
profile_name=inputpath+'p174819.03560'
geomfile_name=inputpath+'g174819.03560'

#************End of User Block*****************
#**********************************************

a=file_IO_obj(profile_type,profile_name,\
                geomfile_type,geomfile_name,\
                outputpath,inputpath)

ne0=a.ne.copy()
x0=a.rhot.copy()


a.modify_profile(q_scale=1.,q_shift=0.,\
                shat_scale=1.,\
                ne_scale=1.2,te_scale=1.,\
                ne_shift=0.,te_shift=0.,\
                Doppler_scale=1.,\
                show_plot=False)
a.output_profile("ITERDB",profile_name+'_mod',shot_num=174819,time_str='3560')

plt.clf()
plt.plot(x0,ne0,alpha=0.7,label='original')
plt.plot(a.rhot,a.ne,alpha=0.7,label='modfied')
plt.legend()
plt.show()
