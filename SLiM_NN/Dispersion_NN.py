# TensorFlow and tf.keras
import tensorflow as tf  #https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
import keras
from sklearn.model_selection import train_test_split

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#****************************************
#**********start of user block***********

filename='./parameter_list_summary.csv'
epochs = 10
batch_size = 100

#**********end of user block*************
#****************************************

#*******start of loading data*******************
df=pd.read_csv(filename)
try:
    df=df.drop(columns=['change'])
except:
    pass

df=df.astype('float')

df_x=pd.DataFrame(np.transpose([df['nu'],df['zeff'],\
                    df['eta'],df['shat'],\
                    df['ky'],df['mu']/df['xstar']]),
                  #index=list('abc'),
                  columns=['nu', 'zeff','eta','shat',\
                    'ky','mu_norm'])
df_y=pd.DataFrame(np.transpose([df['omega_plasma_kHz']/df['omega_n_kHz'],\
                    df['gamma_cs_a']/df['omega_n_cs_a']]),
                  #index=list('abc'),
                  columns=['omega_norm','gamma_norm'])
if testing_output==True:
    print(df_x)
    print(df_y)
    print(len(df_x))
    print(len(df_y))
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

#x=[df['nu'],df['zeff'],df['eta'],df['shat'],df['ky'],df['mu']/df['xstar']]
#y=[df['omega_plasma_kHz']/df['omega_n_kHz'],df['gamma_cs_a']/df['omega_n_cs_a']]

#x_train=x[:int(len(df)*0.8)]
#y_train=y[:int(len(df)*0.8)]

#x_test=x[int(len(df)*0.8):]
#y_test=y[int(len(df)*0.8):]
#*******end of  of loading data*******************


#*********start of creating of model***************

#creating the model
model = keras.Sequential([
    keras.layers.Dense(6),                      # input layer (1)
    keras.layers.Dense(128, activation='relu'), # hidden layer (2)
    keras.layers.Dense(2, activation='softmax') # output layer (3)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#*********end of creating of model***************

#*********start of trainning***********************
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

start = time.time()
with tf.device('/GPU:0'):
    model.fit(x_train, y_train, epochs=epochs)  # we pass the data, labels and epochs and watch the magic!
end = time.time()
print(f"Runtime of the program is {end - start} s")
#save the model

#*********end of trainning***********************

predictions = model.predict(test_images)
