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
plot_profile=False
plot_geofile=True
#************End of User Block*****************
#**********************************************

a=file_IO_obj(profile_type,profile_name,\
                geomfile_type,geomfile_name)

ne0=a.ne.copy()
te0=a.te.copy()
q0=a.q.copy()
x0=a.x.copy()


a.modify_profile(q_scale=2.,q_shift=0.,\
                shat_scale=1.4,\
                ne_scale=1.2,te_scale=1.2,\
                ne_shift=0.,te_shift=0.,\
                Doppler_scale=1.,\
                show_plot=True)

a.output_profile("pfile",profile_name+'_mod',shot_num=174819,time_str='3560')

a.output_geofile("gfile",geomfile_name+'_mod')


b=file_IO_obj("pfile",profile_name+'_mod',\
                "gfile",geomfile_name+'_mod')


if plot_profile==True:
    plt.clf()
    plt.plot(x0,ne0,alpha=0.7,label='original')
    #plt.plot(a.rhot,a.ne,alpha=0.7,label='modfied')
    plt.plot(b.rhot,b.ne,alpha=0.7,label='modfied')
    plt.xlabel('rhot')
    plt.ylabel('ne(/m^3)')
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(x0,te0,alpha=0.7,label='original')
    #plt.plot(a.rhot,a.te,alpha=0.7,label='modfied')
    plt.plot(b.rhot,b.te,alpha=0.7,label='modfied')
    plt.xlabel('rhot')
    plt.ylabel('Te(eV)')
    plt.legend()
    plt.show()

if plot_geofile==True:
    plt.clf()
    plt.plot(x0,q0,alpha=0.7,label='original')
    plt.plot(b.rhot,b.q,alpha=0.7,label='modfied')
    plt.legend()
    plt.show()

