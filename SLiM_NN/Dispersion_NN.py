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

        omega = self.NN_omega_model.predict(np.array([param_omega_norm]),verbose = 0)
        gamma = self.NN_gamma_model.predict(np.array([param_gamma_norm]),verbose = 0)

        [[w]]=omega+gamma*1j

        return w


class Dispersion_NN_beta():
    def __init__(self,\
                 NN_stability_file,\
                 NN_omega_file,\
                 NN_gamma_file,\
                norm_stability_csv_file,\
                norm_omega_csv_file,\
                norm_gamma_csv_file):

        NN_stability_model = tf.keras.models.load_model(NN_stability_file)
        NN_omega_model = tf.keras.models.load_model(NN_omega_file)
        NN_gamma_model = tf.keras.models.load_model(NN_gamma_file)
        
        
        NN_stability_model.summary()
        NN_omega_model.summary()
        NN_gamma_model.summary()

        df_stability=pd.read_csv(norm_stability_csv_file)
        df_omega=pd.read_csv(norm_omega_csv_file)
        df_gamma=pd.read_csv(norm_gamma_csv_file)

        self.NN_stability_file=NN_stability_file
        self.NN_omega_file=NN_omega_file
        self.NN_gamma_file=NN_gamma_file

        self.norm_stability_csv_file=norm_stability_csv_file
        self.norm_omega_csv_file=norm_omega_csv_file
        self.norm_gamma_csv_file=norm_gamma_csv_file

        self.NN_stability_model=NN_stability_model
        self.NN_omega_model=NN_omega_model
        self.NN_gamma_model=NN_gamma_model

        self.norm_stability_factor=df_stability['factor']
        self.norm_omega_factor=df_omega['factor']
        self.norm_gamma_factor=df_gamma['factor']



    def Dispersion_omega(self,nu,zeff,eta,shat,beta,ky,mu,xstar):
        param=[nu,zeff,eta,shat,beta,ky,mu/xstar]
        param_omega_norm=np.zeros(7,dtype=float)
        param_gamma_norm=np.zeros(7,dtype=float)

        for i in range(7):
            param_omega_norm[i]=param[i]*self.norm_omega_factor[i]
            param_gamma_norm[i]=param[i]*self.norm_gamma_factor[i]

        [[stability]] = self.NN_omega_model.predict(np.array([param_omega_norm]),verbose = 0)
        omega = self.NN_omega_model.predict(np.array([param_omega_norm]),verbose = 0)

        if stability>0.5:
            gamma = self.NN_gamma_model.predict(np.array([param_gamma_norm]),verbose = 0)
            print('\ngamma='+str(gamma))
        else:
            gamma=0.

        [[w]]=omega+gamma*1j

        return w

        


class Dispersion_NN_beta_2():
    def __init__(self,NN_file,norm_csv_file):
        NN_model = tf.keras.models.load_model(NN_file)
        
        NN_model.summary()

        df=pd.read_csv(norm_csv_file)


        self.NN_file=NN_file
        self.norm_csv_file=norm_csv_file

        self.NN_model=NN_model

        self.norm_factor=df['factor']

    def Dispersion_omega(self,nu,zeff,eta,shat,beta,ky,mu,xstar):
        param=[nu,zeff,eta,shat,beta,ky,mu/xstar]
        param_norm=np.zeros(7,dtype=float)

        for i in range(7):
            param_norm[i]=param[i]*self.norm_factor[i]

        [[omega,gamma]] = self.NN_model.predict(np.array([param_norm]),verbose = 0)
        
        w=omega+gamma*1j

        return w
