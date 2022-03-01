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
filename='./NN_data/0MTM_scan_CORI_2.csv'
epochs = 150
batch_size = 100
checkpoint_path='./tmp/checkpoint'
#**********end of user block*************
#****************************************

#*********start of creating of model***************
def create_model(checkpoint_path):
    #creating the model
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(6))) # input layer (1)
    model.add(tf.keras.layers.Dense(units=16, activation='relu')) # input layer (1)
    model.add(tf.keras.layers.Dense(units=32, activation='relu')) # input layer (1)
    #model.add(tf.keras.layers.Dense(units=256, activation='relu')) # input layer (1)
    model.add(tf.keras.layers.Dropout(0.2)) # input layer (1)
    model.add(tf.keras.layers.Dense(units=4, activation='relu')) # input layer (1)
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # input layer (1)
    model.summary()

    model.compile(loss='binary_crossentropy',\
                #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),\
                metrics=['accuracy'])

    #*create callback function (optional)
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,log={}):
    
            #print(log.get.keys())
            #print(log.get('epoch'))
            if(log.get('val_accuracy')>0.99):
                print('val_accuracy>0.99, stop training!')
                self.model.stop_training=True
    
    callbacks=myCallback()
    
    import os 
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    #https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch')
    callback_func=[cp_callback,callbacks]

    #*********end of creating of model***************
    return model,callback_func



def load_data(filename):
    #*******start of loading data*******************
    
    df=pd.read_csv(filename)
    df=df.dropna()
    try:
        df=df.drop(columns=['change'])
    except:
        pass
    
    #df=df.astype('float')
    
    df_unstable=df.query('omega_omega_n!=0 and gamma_omega_n>0')
    df_stable=df.query('omega_omega_n==0 or gamma_omega_n<=0')
    
    
    df_unstable['unstable']=[1]*len(df_unstable)
    
    df_stable['unstable']=[0]*len(df_stable)
    
    df=pd.concat([df_unstable, df_stable], axis=0)
    
    df_x=pd.DataFrame(np.transpose([df['nu'],df['zeff'],\
                        df['eta'],df['shat'],\
                        df['ky'],df['mu']/df['xstar']]),
                      columns=['nu', 'zeff','eta','shat',\
                        'ky','mu_norm'])

    keys=df_x.keys()
    df_norm_name=[i for i in keys]
    df_norm_factor=[1./np.max(df_x[i]) for i in keys]
    
    
    for i in range(len(keys)):
        df_x[keys[i]]=df_x[keys[i]]*df_norm_factor[i]
    
    
    #df_y=pd.DataFrame(np.transpose([df['unstable'],df['stable']]),
    #                  columns=['unstable','stable'])
    
    df_y=pd.DataFrame(np.transpose([df['unstable']]),\
                      columns=['unstable'])
    df_y=df_y.astype('int32')
    #print(df)
    #input()
    d = {'name':df_norm_name,'factor':df_norm_factor}
    df=pd.DataFrame(d, columns=['name','factor'])   #construct the panda dataframe
    df.to_csv('NN_norm_factor.csv',index=False)

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)
    
    #*******end of  of loading data*******************
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test=load_data(filename)

#*********start of trainning***********************
#print(x_test)
#print(y_test)
#input()
model,callback_func=create_model(checkpoint_path)

model.load_weights(checkpoint_path)

history=model.fit(x_train, y_train, epochs=epochs,
            callbacks=callback_func,\
            validation_data=(x_test,y_test))  

#save the model
model.save("SLiM_NN_stabel_unstable.h5")  # we can save the model and reload it at anytime in the future
#*********end of trainning***********************

from Post_plot_learning_rate import plot_hist
plot_hist(history)

