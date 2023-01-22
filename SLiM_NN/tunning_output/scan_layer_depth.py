from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="out_layer_depth.txt"
hyper_paramerter_name_list=['num_layers']
hyper_paramerter_dtype_list=['int']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['num_layers'],df['Score'])
plt.xlabel('num_layers')
plt.ylabel('Score')
plt.show()