import matplotlib.pyplot as plt

from SLiM_phys.SLiM_obj import mode_finder

#**********************************************
#**********Start of User block*****************
profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
outputpath='./../Test_files/'
path='./../Test_files/'
profile_name=path+'p174819.03560'
geomfile_name=path+'g174819.03560'
x0_min=0.93         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor
#************End of User Block*****************
#**********************************************

mode_finder_obj=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)


mode_finder_obj.Plot_q_scale_rational_surfaces_red_and_green(peak_percent=0.06,\
            q_scale=0.98,q_shift=0.0,q_uncertainty=0.1,\
            n_list=[3,4,5],unstable_list=[1,0,1])
