# TensorFlow and tf.keras
import tensorflow as tf  #https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
import keras
from sklearn.model_selection import train_test_split

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#**********start of user block***********
filename='./NN_data/0MTM_scan_CORI_2.csv'
loaded_model = tf.keras.models.load_model('./Trained_model/SLiM_NN_omega.h5')
#**********end of user block*************
#****************************************

df=pd.read_csv(filename)
try:
    df=df.drop(columns=['change'])
except:
    pass

df=df.astype('float')
df=df.query('omega_omega_n!=0 and gamma_omega_n>0')
df_x=pd.DataFrame(np.transpose([df['nu'],df['zeff'],\
                    df['eta'],df['shat'],\
                    df['ky'],df['mu']/df['xstar']]),
                  columns=['nu', 'zeff','eta','shat',\
                    'ky','mu_norm'])
df_y=pd.DataFrame(np.transpose([df['omega_omega_n']]),\
                  columns=['omega_norm'])

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5)
#*********end of trainning***********************
loaded_model.summary()
print(df_x)
predictions = loaded_model.predict(df_x)

print('df_x')
print(df_x)
print('df_y')
print(df_y)
print('predictions')
print(predictions)
print('abs(df_y-predictions)')
print(np.mean(abs(df_y-predictions)/df_y))
print('std(df_y-predictions)')
print(np.std(abs(df_y-predictions)/df_y))