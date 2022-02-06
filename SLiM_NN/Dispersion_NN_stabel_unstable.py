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
filename='./NN_data/0MTM_scan.csv'
epochs = 10
batch_size = 100
testing_output=True
#**********end of user block*************
#****************************************

#*********start of creating of model***************
def create_model():
    #creating the model
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(6))) # input layer (1)
    model.add(tf.keras.layers.Dense(units=32, activation='relu')) # input layer (1)
    model.add(tf.keras.layers.Dense(units=64, activation='relu')) # input layer (1)
    model.add(tf.keras.layers.Dense(units=32, activation='relu')) # input layer (1)
    model.add(tf.keras.layers.Dense(units=2, activation='relu')) # input layer (1)

    print(model.summary())
    
    model.compile(loss='binary_crossentropy',\
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),\
                    )
    #*********end of creating of model***************
    return model

#*******start of loading data*******************
df=pd.read_csv(filename)
try:
    df=df.drop(columns=['change'])
except:
    pass

df=df.astype('float')

df_unstable=df.query('omega_omega_n!=0 and gamma_omega_n>0')
df_stable=df.query('omega_omega_n==0 or gamma_omega_n<=0')

df_unstable['unstable']=[1]*len(df_unstable)
df_unstable['unstable']=[0]*len(df_unstable)
df_stable['unstable']=[0]*len(df_stable)
df_stable['stable']=[1]*len(df_stable)

df=pd.concat([df_unstable, df_stable], axis=0)
print(df)

df_x=pd.DataFrame(np.transpose([df['nu'],df['zeff'],\
                    df['eta'],df['shat'],\
                    df['ky'],df['mu']/df['xstar']]),
                  columns=['nu', 'zeff','eta','shat',\
                    'ky','mu_norm'])

df_y=pd.DataFrame(np.transpose([df['unstable'],df['stable']]),
                  columns=['unstable','stable'])

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

if testing_output==True:
    print('x_train')
    print(x_train)
    print('y_train')
    print(y_train)

#*******end of  of loading data*******************

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


#*********start of trainning***********************
start = time.time()
#with tf.device('/GPU:0'):
model=create_model()
print(np.shape(x_train))
print(np.shape(y_train))
model.fit(x_train, y_train, epochs=epochs)  # we pass the data, labels and epochs and watch the magic!
end = time.time()
print(f"Runtime of the program is {end - start} s")
#save the model

#the trained model can be saved 
model.save("SLiM_NN_stable.h5")  # we can save the model and reload it at anytime in the future
load_model = tf.keras.models.load_model('SLiM_NN_stable.h5')


#*********end of trainning***********************
predictions = load_model.predict(x_train)
print('x_train')
print(x_train)
print('y_train')
print(y_train)
print('predictions')
print(predictions)
print('abs(y_train-predictions)')
print(abs(y_train-predictions))

predictions = load_model.predict(x_test)
print('y_test')
print(y_test)
print('predictions')
print(predictions)
print('abs(y_train-predictions)')
print(abs(y_test-predictions))
