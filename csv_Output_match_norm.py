import pandas as pd

intput_para='./Test_files/Output_test/parameter_list.csv'
intput_calc='./Test_files/Output_test/0Disperson_calc.csv'
output_name='./Test_files/Output_test/joint_and_normed.csv'

mode=1 	#mode1 is for the results from the MPI calculations(has normalized frequency and growth rate)
		#mode2 is for the results calculated from PC(has frequency in kHz and growth in cs/a)

df_para=pd.read_csv(intput_para) 
df_calc=pd.read_csv(intput_calc)

df_para_key=df_para.keys()
df_calc_key=df_para.keys()



for i in range(len(df_para)):
	for j in range(len(df_calc)):
		if df_para['nu'][i]==df_calc['nu'][j]\
			and df_para['zeff'][i]==df_calc['zeff'][j]\
			and df_para['eta'][i]==df_calc['eta'][j]\
			and df_para['ky'][i]==df_calc['ky'][j]\
			and df_para['mu'][i]==df_calc['mu'][j]\
			and df_para['xstar'][i]==df_calc['xstar'][j]\
			and df_para['ModIndex'][i]==df_calc['ModIndex'][j]:
			if mode==1:
				df_para['omega_plasma_kHz'][i]=df_calc['omega_omega_n'][j]*df_para['omega_n_kHz'][i]
				df_para['gamma_cs_a'][i]=df_calc['gamma_omega_n'][j]*df_para['omega_n_cs_a'][i]
				df_para['omega_lab_kHz'][i]=df_para['omega_plasma_kHz'][i]\
											-df_para['omega_e_plasma_kHz'][i]\
											+df_para['omega_e_lab_kHz'][i]
			elif mode==2:
				df_para['omega_plasma_kHz'][i]=df_calc['omega_plasma_kHz'][j]
				df_para['omega_lab_kHz'][i]=df_calc['omega_lab_kHz'][j]
				df_para['gamma_cs_a'][i]=df_calc['gamma_cs_a'][j]
				


df_out=pd.DataFrame(df_para, columns=df_para.keys())	#construct the panda dataframe
df_out.to_csv(output_name,index=False)