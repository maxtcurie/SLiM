from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="out_activation.txt"
hyper_paramerter_name_list=['activation0','activation1','activation2']
hyper_paramerter_dtype_list=['string','string','string']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['activation0'],df['Score'])
plt.xlabel('activation0')
plt.ylabel('Score')
plt.show()