import tensorflow as tf  
import pandas as pd 
import numpy as np 

class Dispersion_NN():
    def __init__(self,NN_omega_file,NN_gamma_file,\
                norm_omega_csv_file,norm_gamma_csv_file):
        NN_omega_model = tf.keras.models.load_model(NN_omega_file)
        NN_gamma_model = tf.keras.models.load_model(NN_gamma_file)
        
        NN_omega_model.summary()
        NN_gamma_model.summary()

        df_omega=pd.read_csv(norm_omega_csv_file)
        df_gamma=pd.read_csv(norm_gamma_csv_file)

        self.NN_omega_file=NN_omega_file
        self.NN_gamma_file=NN_gamma_file
        self.norm_omega_csv_file=norm_omega_csv_file
        self.norm_gamma_csv_file=norm_gamma_csv_file

        self.NN_omega_model=NN_omega_model
        self.NN_gamma_model=NN_gamma_model
        self.norm_omega_factor=df_omega['factor']
        self.norm_gamma_factor=df_gamma['factor']

    def Dispersion_omega(self,nu,zeff,eta,shat,beta,ky,mu,xstar):
        param=[nu,zeff,eta,shat,beta,ky,mu/xstar]
        param_omega_norm=np.zeros(7,dtype=float)
        param_gamma_norm=np.zeros(7,dtype=float)

        for i in range(7):
            param_omega_norm[i]=param[i]*self.norm_omega_factor[i]
            param_gamma_norm[i]=param[i]*self.norm_gamma_factor[i]

        omega = self.NN_omega_model.predict(np.array([param_omega_norm]))
        gamma = self.NN_gamma_model.predict(np.array([param_gamma_norm]))

        [[w]]=omega+gamma*1j

        return w
