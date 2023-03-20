from SLiM_phys.SLiM_obj import mode_finder

#**********************************************
#**********Start of User block*****************
profile_type= "ITERDB"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
outputpath='./../Test_files/'
path='./../Test_files/'
profile_name=path+'DIIID174819.iterdb'
geomfile_name=path+'g174819.03560'
x0_min=0.93         # beginning of the range in rho_tor
x0_max=0.99         # ending of the range in rho_tor
#************End of User Block*****************
#**********************************************

mode_finder_obj=mode_finder(profile_type,profile_name,\
            geomfile_type,geomfile_name,\
            outputpath,path,x0_min,x0_max)

q_scale=0.98
q_shift=0.
shat_scale=1.
ne_scale=1
te_scale=1.
ne_shift=0.
te_shift=0.
Doppler_scale=1

mode_finder_obj.modify_profile(\
    q_scale=q_scale,q_shift=q_shift,\
    shat_scale=shat_scale,\
    ne_scale=ne_scale,te_scale=te_scale,\
    ne_shift=ne_shift,te_shift=te_shift,\
    Doppler_scale=Doppler_scale,\
    show_plot=False)

mode_finder_obj.reset_profile() #reset the profile back to nominal


