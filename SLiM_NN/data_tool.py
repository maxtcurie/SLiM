import tensorflow as tf  
import pandas as pd 
import numpy as np 

def load_data(filename_list):
    #*******start of loading data*******************
    for i in range(len(filename_list)):
        filename=filename_list[i]
        df=pd.read_csv(filename)
        df=df.dropna()
        try:
            df=df.drop(columns=['change'])
        except:
            pass

        #merge the dataframe
        if i==0:
            df_merge=df
        elif i!=0:
            df_merge=pd.concat([df_merge, df], axis=0)
            
    return df_merge
def find_data_range(para_list_min,para_list_max):
    para_name_list=['nu', 'zeff', 'eta', 'shat',  'beta',  'ky',   'mu', 'xstar']
    query_name_str=''

    for i in range(len(para_name_list)):
        if i==0:
            query_name_str=query_name_str+\
                            str(para_list_min[i])+'<'+para_name_list[i]+\
                            ' and '+\
                            para_name_list[i]+'<'+str(para_list_max[i])
        else:
            query_name_str=query_name_str+' and '+\
                            str(para_list_min[i])+'<'+para_name_list[i]+\
                            ' and '+\
                            para_name_list[i]+'<'+str(para_list_max[i])

    return query_name_str
