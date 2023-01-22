from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="out_learning_rate.txt"
hyper_paramerter_name_list=['learning_rate']
hyper_paramerter_dtype_list=['float']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['learning_rate'],df['Score'])
plt.xlabel('learning_rate')
plt.ylabel('Score')
plt.xscale('log')
plt.show()