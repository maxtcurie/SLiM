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

mode_finder_obj.std_SLiM_calc(n_min=1,n_max=5,surface_num=1,\
              Run_mode=2,peak_percent=0.2,\
              manual_fit=True,\
              NN_path='./SLiM_NN/Trained_model')