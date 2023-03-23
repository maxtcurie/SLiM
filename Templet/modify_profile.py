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
mode_finder_obj.modify_profile(q_scale=q_scale,q_shift=0.,\
                                    shat_scale=1.2,\
                                    ne_scale=1.2,te_scale=0.8,\
                                    ne_shift=0.,te_shift=0.,\
                                    Doppler_scale=1.2,\
                                    show_plot=True)
mode_finder_obj.reset_profile() #reset the profile back to nominal