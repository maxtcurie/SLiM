import tensorflow as tf  
import pandas as pd 
import numpy as np 


#stability, omega 2 net
class Dispersion_NN():
    def __init__(self,NN_path):
        if NN_path[-1]!='/':
            NN_path=NN_path+'/'

        self.NN_stability_file=NN_path+'SLiM_NN_stability.h5'
        self.NN_omega_file=NN_path+'SLiM_NN_omega.h5'

        self.norm_stability_csv_file=NN_path+'NN_stability_norm_factor.csv'
        self.norm_omega_csv_file=NN_path+'NN_omega_norm_factor.csv'

        NN_stability_model = tf.keras.models.load_model(self.NN_stability_file)
        NN_omega_model = tf.keras.models.load_model(self.NN_omega_file)
        
        NN_stability_model.summary()
        NN_omega_model.summary()

        df_stability=pd.read_csv(self.norm_stability_csv_file)
        df_omega=pd.read_csv(self.norm_omega_csv_file)
        
        self.NN_stability_model=NN_stability_model
        self.NN_omega_model=NN_omega_model
        
        self.norm_stability_factor=df_stability
        self.norm_omega_factor=df_omega



    def Dispersion_omega(self,nu,zeff,eta,shat,beta,ky,mu,xstar):
        param=np.array([nu,zeff,eta,shat,beta,ky,mu/xstar])
        param_omega_norm=np.zeros(7,dtype=float)
        param_stability_norm=np.zeros(7,dtype=float)
                  #nu, zeff, eta, shat, beta, ky, mu/xstar  gamma
        log_scale=[1,  0,    0,   1,    1,    0,  0       , 0    ]

        for i in range(7):
            if log_scale[i]==1:
                param_temp=np.log(param[i])
            else:
                param_temp=param[i]
            param_stability_norm[i]=(param_temp-self.norm_stability_factor['offset'][i])*self.norm_stability_factor['factor'][i]
            param_omega_norm[i]    =(param_temp-self.norm_omega_factor['offset'][i])    *self.norm_omega_factor['factor'][i]
            

        [[stability]] = self.NN_stability_model.predict(np.array([param_stability_norm]),verbose = 0)
        

        if stability>0.2:
            [[omega]] = self.NN_omega_model.predict(np.array([param_omega_norm]),verbose = 0)
            gamma=1.
        else:
            gamma=0.
            omega=0.

        #print(self.norm_omega_factor)

        omega=omega/(self.norm_omega_factor['factor'][7])+(self.norm_omega_factor['offset'][7])
        w=omega+gamma*1j
        
        return w


#gamma, omega 3 net
class Dispersion_NN_beta3():
    def __init__(self,NN_path):
        if NN_path[-1]!='/':
            NN_path=NN_path+'/'

        self.NN_stability_file=NN_path+'SLiM_NN_stability.h5'
        self.NN_omega_file=NN_path+'SLiM_NN_omega.h5'
        self.NN_gamma_file=NN_path+'SLiM_NN_gamma.h5'

        self.norm_stability_csv_file=NN_path+'NN_stability_norm_factor.csv'
        self.norm_omega_csv_file=NN_path+'NN_omega_norm_factor.csv'
        self.norm_gamma_csv_file=NN_path+'NN_gamma_norm_factor.csv'

        NN_stability_model = tf.keras.models.load_model(self.NN_stability_file)
        NN_omega_model = tf.keras.models.load_model(self.NN_omega_file)
        NN_gamma_model = tf.keras.models.load_model(self.NN_gamma_file)
        
        
        NN_stability_model.summary()
        NN_omega_model.summary()
        NN_gamma_model.summary()

        df_stability=pd.read_csv(self.norm_stability_csv_file)
        df_omega=pd.read_csv(self.norm_omega_csv_file)
        df_gamma=pd.read_csv(self.norm_gamma_csv_file)

        self.NN_stability_model=NN_stability_model
        self.NN_omega_model=NN_omega_model
        self.NN_gamma_model=NN_gamma_model

        self.norm_stability_factor=df_stability
        self.norm_omega_factor=df_omega
        self.norm_gamma_factor=df_gamma



    def Dispersion_omega(self,nu,zeff,eta,shat,beta,ky,mu,xstar):
        param=np.array([nu,zeff,eta,shat,beta,ky,mu/xstar])
        param_omega_norm=np.zeros(7,dtype=float)
        param_gamma_norm=np.zeros(7,dtype=float)
        param_stability_norm=np.zeros(7,dtype=float)
                  #nu, zeff, eta, shat, beta, ky, mu/xstar  gamma
        log_scale=[1,  0,    0,   1,    1,    0,  0       , 0    ]

        for i in range(7):
            if log_scale[i]==1:
                param_temp=np.log(param[i])
            else:
                param_temp=param[i]
            param_stability_norm[i]=(param_temp-self.norm_stability_factor['offset'][i])*self.norm_stability_factor['factor'][i]
            param_omega_norm[i]    =(param_temp-self.norm_omega_factor['offset'][i])    *self.norm_omega_factor['factor'][i]
            param_gamma_norm[i]    =(param_temp-self.norm_gamma_factor['offset'][i])    *self.norm_gamma_factor['factor'][i]

        [[stability]] = self.NN_stability_model.predict(np.array([param_stability_norm]),verbose = 0)
        

        if stability>0.5:
            gamma = self.NN_gamma_model.predict(np.array([param_gamma_norm]),verbose = 0)
            omega = self.NN_omega_model.predict(np.array([param_omega_norm]),verbose = 0)
            print('\ngamma='+str(gamma))
            gamma=1.
        else:
            gamma=0.
            omega=0.

        #print(self.norm_omega_factor)

        omega=omega/(self.norm_omega_factor['factor'][7])+(self.norm_omega_factor['offset'][7])
        gamma=gamma/(self.norm_gamma_factor['factor'][7])+(self.norm_gamma_factor['offset'][7])
        w=omega+gamma*1j
        
        return w


        
#gamma, omega 2 net
class Dispersion_NN_beta():
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



#gamma, omega 1 net
class Dispersion_NN_beta_2():
    def __init__(self,NN_file,norm_csv_file):
        NN_model = tf.keras.models.load_model(NN_file)
        
        NN_model.summary()

        df=pd.read_csv(norm_csv_file)


        self.NN_file=NN_file
        self.norm_csv_file=norm_csv_file
        self.NN_model=NN_model
        self.norm_factor=df


    def Dispersion_omega(self,nu,zeff,eta,shat,beta,ky,mu,xstar):
        param=[nu,zeff,eta,shat,beta,ky,mu/xstar]
        param_norm=np.zeros(7,dtype=float)

                  #nu, zeff, eta, shat, beta, ky, mu/xstar  gamma omega 
        log_scale=[1,  0,    0,   1,    1,    0,  0       , 0    , 0   ]

        for i in range(7):
            if log_scale[i]==1:
                param_temp=np.log(param[i])
            else:
                param_temp=param[i]
            param_norm[i]=(param_temp-self.norm_factor['offset'][i])*self.norm_factor['factor'][i]
            

        [[omega,gamma]] = self.NN_model.predict(np.array([param_norm]),verbose = 0)
        

        omega=omega/(self.norm_factor['factor'][7])+(self.norm_factor['offset'][7])
        gamma=gamma/(self.norm_factor['factor'][8])+(self.norm_factor['offset'][8])

        w=omega+gamma*1j

        return w
