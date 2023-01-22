from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="out_optimizers.txt"
hyper_paramerter_name_list=['optimizers']
hyper_paramerter_dtype_list=['string']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['optimizers'],df['Score'])
plt.xlabel('optimizers')
plt.ylabel('Score')
plt.show()