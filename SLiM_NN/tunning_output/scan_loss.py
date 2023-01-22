from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="loss_scan_out.txt"
hyper_paramerter_name_list=['loss']
hyper_paramerter_dtype_list=['string']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['loss'],df['Score'])
plt.xlabel('loss')
plt.ylabel('Score')
plt.show()