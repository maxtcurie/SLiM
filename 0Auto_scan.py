from glob import glob
import os
import pandas as pd
import sys
sys.path.insert(1, './Tools')
from SLiM_obj import mode_finder 
from file_IO_obj import dir_scaner
from file_IO_obj import get_g_and_p_file

#this script read all the readable p and g file in the folder and do basic SLiM analysis
#the default p and g form is 

base_dir='./'
run_dir_list=dir_scaner(base_dir,level=1)

Run_mode=6      # mode1: fast mode
                # mode2: slow mode(global)
                # mode3: slow mode(local) 
                # mode4: slow mode manual(global)
                # mode5: slow slow mode(global)
                # mode6: NN mode (global)

#run_dir_list=run_dir_list[:3]

df={}
df['pfile']=[]
df['gfile']=[]
df['working']=[]

df['r_peak']=[]
df['nu']=[]
df['shat']=[]
df['beta']=[]
df['beta_hat']=[]


for dir_name in run_dir_list:
	print('****************')
	gfile,pfile=get_g_and_p_file(dir_name)
	if gfile=='':
		for key in df.keys():
			df[key].append(0)
		continue 
	else:
		print(gfile,pfile)
		df['working'].append(1)

	df['pfile'].append(pfile)
	df['gfile'].append(gfile)
	

	mode_finder_obj=mode_finder('pfile',pfile,'gfile',gfile,\
					dir_name,dir_name,0.8,0.99)
	mode_finder_obj.ome_peak_range(0.1)

	r_peak=mode_finder_obj.x_peak

	nu,zeff,eta,shat,beta,ky,mu,xstar=\
                    mode_finder_obj.parameter_for_dispersion(r_peak,1)
	df['r_peak'].append(r_peak)
	df['nu'].append(nu)
	df['shat'].append(shat)
	df['beta'].append(beta)
	df['beta_hat'].append(beta/shat**2.)

	df_calc,df_para=mode_finder_obj.std_SLiM_calc(Run_mode,NN_path='../../SLiM/SLiM_NN/Trained_model')
	df_calc.to_csv(dir_name+'/parameter.csv')

df=pd.DataFrame(df)	
df.to_csv('./output.csv')