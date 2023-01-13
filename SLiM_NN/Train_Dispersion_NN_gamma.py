# TensorFlow and tf.keras
import tensorflow as tf  #https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
#import keras
from sklearn.model_selection import train_test_split

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from file_list import file_list

#****************************************
#**********start of user block***********
filename_list=file_list()
epochs = 1
batch_size = 100
checkpoint_path='./tmp/checkpoint'
Read_from_checkpoint=True
#**********end of user block*************
#****************************************

#*********start of creating of model***************
def create_model(checkpoint_path):
    #creating the model
    
    model = tf.keras.Sequential([
                    tf.keras.Input(shape=(7)),
                    tf.keras.layers.Dense(units=64, activation='relu'),
                    tf.keras.layers.Dense(units=128, activation='relu'),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(units=512, activation='relu'),
                    tf.keras.layers.Dense(units=128, activation='relu'),
                    tf.keras.layers.Dense(units=64, activation='relu'),
                    tf.keras.layers.Dense(units=32, activation='relu'),
                    tf.keras.layers.Dense(units=8, activation='relu'),
                    tf.keras.layers.Dense(units=1, activation='relu')
        ])


    #model.summary()

    model.compile(loss='MeanSquaredLogarithmicError',\
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                #optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),\
                metrics=['mean_absolute_error'])

    #*create callback function (optional)
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,log={}):
    
            #print(log.get.keys())
            #print(log.get('epoch'))
            if(log.get('mean_absolute_error')<0.0001):
                print('mean_absolute_error<0.0001, stop training!')
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

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
    )
    callback_func=[cp_callback,callbacks,lr_callback]

    #*********end of creating of model***************
    return model,callback_func



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
        
        
        df=df.query('omega_omega_n!=0 and gamma_omega_n>0')
        #df_stable=df.query('omega_omega_n==0 or gamma_omega_n<=0')
        
        
        #df_unstable['unstable']=[1]*len(df_unstable)
        #df_stable['unstable']=[0]*len(df_stable)
        
        #df=pd.concat([df_unstable, df_stable], axis=0)
        
        df_x=pd.DataFrame(np.transpose([df['nu'],df['zeff'],\
                            df['eta'],df['shat'],df['beta'],\
                            df['ky'],df['mu']/df['xstar']]),
                          columns=['nu', 'zeff','eta','shat',\
                                    'beta','ky','mu_norm'])

        df_y=pd.DataFrame(np.transpose([df['gamma_omega_n']]),\
                              columns=['gamma_omega_n'])
        df_y=df_y.astype('float')

        #merge the dataframe
        if i==0:
            df_x_merge=df_x
            df_y_merge=df_y
        elif i!=0:
            df_x_merge=pd.concat([df_x_merge, df_x], axis=0)
            df_y_merge=pd.concat([df_y_merge, df_y], axis=0)

    print(df_y_merge[:10])

    #get normalizing factor
    keys_x=df_x_merge.keys()
              #nu, zeff, eta, shat, beta, ky, mu/xstar  gamma
    log_scale=[1,  0,    0,   1,    1,    0,  0       , 0    ]
    df_x_norm_name=[i for i in keys_x]

    df_x_after_log={}
    for i in range(len(keys_x)):
        if log_scale[i]==1:
            df_x_after_log[keys_x[i]]=np.log(df_x_merge[keys_x[i]])
        else:
            df_x_after_log[keys_x[i]]=df_x_merge[keys_x[i]]

    df_x_norm_offset=[np.min(df_x_after_log[i]) for i in keys_x]
    df_x_norm_factor=[1./(np.max(df_x_after_log[i])-np.min(df_x_after_log[i])) for i in keys_x]
    

    keys_y=df_y_merge.keys()
    df_y_norm_offset=[np.min(df_y_merge[i]) for i in keys_y]
    df_y_norm_factor=[1./(np.max(df_y_merge[i])-np.min(df_y_merge[i])) for i in keys_y]

    df_norm_name=[]
    df_norm_factor=[]
    df_norm_offset=[]
    for i in range(len(keys_x)):
        df_norm_name.append(keys_x[i])
        df_norm_factor.append(df_x_norm_factor[i])
        df_norm_offset.append(df_x_norm_offset[i])

    for i in range(len(keys_y)):
        df_norm_name.append(keys_y[i])
        df_norm_factor.append(df_y_norm_factor[i])
        df_norm_offset.append(df_y_norm_offset[i])

    d = {'name':df_norm_name,'factor':df_norm_factor,\
        'offset':df_norm_offset,'log':log_scale}
    df_norm=pd.DataFrame(d, columns=['name','factor','offset','log'])   #construct the panda dataframe
    df_norm.to_csv('./Trained_model/NN_gamma_norm_factor.csv',index=False)
    
    #print(df_norm)

    df_x_after_norm={}
    df_y_after_norm={}
    for i in range(len(keys_x)):
        df_x_after_norm[keys_x[i]]=(df_x_after_log[keys_x[i]]-df_x_norm_offset[i])*df_x_norm_factor[i]

    for i in range(len(keys_y)):
        df_y_after_norm[keys_y[i]]=(df_y_merge[keys_y[i]]    -df_y_norm_offset[i])*df_y_norm_factor[i]
    
    df_x_after_norm=pd.DataFrame(df_x_after_norm, columns=keys_x)
    df_y_after_norm=pd.DataFrame(df_y_after_norm, columns=keys_y)

    x_train, x_test, y_train, y_test = train_test_split(df_x_after_norm, df_y_after_norm, test_size=0.1)
        
    #*******end of  of loading data*******************
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test=load_data(filename_list)

#*********start of trainning***********************
print('x_test')
print(x_test)
print(y_test)
print(len(x_test)+len(x_train))
#input()
model,callback_func=create_model(checkpoint_path)
if Read_from_checkpoint:
    model.load_weights(checkpoint_path)
history=model.fit(x_train, y_train, epochs=epochs,
            callbacks=callback_func,\
            validation_data=(x_test,y_test))  

#save the model
model.save("./Trained_model/SLiM_NN_gamma.h5")  # we can save the model and reload it at anytime in the future
#*********end of trainning***********************

from Post_plot_learning_rate import plot_hist
plot_hist(history)
