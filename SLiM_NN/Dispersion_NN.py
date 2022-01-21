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

filename='./0MTM_scan_2022_01_19.csv'
epochs = 10
batch_size = 100
testing_output=False
#**********end of user block*************
#****************************************

#*******start of loading data*******************
df=pd.read_csv(filename)
try:
    df=df.drop(columns=['change'])
except:
    pass

df=df.astype('float')

df=df.query('omega_omega_n!=0')

df_x=pd.DataFrame(np.transpose([df['nu'],df['zeff'],\
                    df['eta'],df['shat'],\
                    df['ky'],df['mu']/df['xstar']]),
                  #index=list('abc'),
                  columns=['nu', 'zeff','eta','shat',\
                    'ky','mu_norm'])
df_y=pd.DataFrame(np.transpose([df['omega_omega_n'],\
                    df['gamma_omega_n']]),
                  #index=list('abc'),
                  columns=['omega_norm','gamma_norm'])

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

if testing_output==True:
    print(df_x)
    print(df_y)
    print(len(df_x))
    print(len(df_y))
    print(x_train)

#x=[df['nu'],df['zeff'],df['eta'],df['shat'],df['ky'],df['mu']/df['xstar']]
#y=[df['omega_plasma_kHz']/df['omega_n_kHz'],df['gamma_cs_a']/df['omega_n_cs_a']]

#x_train=x[:int(len(df)*0.8)]
#y_train=y[:int(len(df)*0.8)]

#x_test=x[int(len(df)*0.8):]
#y_test=y[int(len(df)*0.8):]
#*******end of  of loading data*******************


#*********start of creating of model***************

#creating the model
'''
model = keras.Sequential([
    keras.layers.Dense(input_shape=(6)),                      # input layer (1)
    keras.layers.Dense(64, activation='relu'), # hidden layer (2)
    keras.layers.Dense(64, activation='relu'), # hidden layer (3)
    keras.layers.Dense(2, activation='softmax') # output layer (4)
])
'''
gpus=tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model = keras.Sequential([])
model.add(tf.keras.Input(shape=(6,)))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(2))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(len(model.weights))
#*********end of creating of model***************

#*********start of trainning***********************
start = time.time()
#with tf.device('/GPU:0'):
model.fit(x_train, y_train, epochs=epochs)  # we pass the data, labels and epochs and watch the magic!
end = time.time()
print(f"Runtime of the program is {end - start} s")
#save the model

#*********end of trainning***********************

predictions = model.predict(x_train)
