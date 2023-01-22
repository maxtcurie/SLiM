from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="out_layer_node.txt"
hyper_paramerter_name_list=['layer_nodes']
hyper_paramerter_dtype_list=['int']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['layer_nodes'],df['Score'])
plt.xlabel('layer_nodes')
plt.ylabel('Score')
plt.xscale('log')
plt.show()