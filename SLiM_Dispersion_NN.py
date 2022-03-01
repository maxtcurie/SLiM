from SLiM_NN.Dispersion_NN import Dispersion_NN

path='./SLiM_NN/Trained_model/'
NN_omega_file	   =path+'SLiM_NN_omega.h5'
NN_gamma_file	   =path+'SLiM_NN_stabel_unstable.h5'
norm_omega_csv_file=path+'NN_omega_norm_factor.csv'
norm_gamma_csv_file=path+'NN_gamma_norm_factor.csv'
[nu, zeff, eta, shat,  beta,  ky,   mu, xstar]=\
[1., 1.5,  1.5, 0.001, 0.001, 0.03, 0., 10.  ]

Dispersion_NN_obj=Dispersion_NN(NN_omega_file,NN_gamma_file,norm_omega_csv_file,norm_gamma_csv_file)
w=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar)

print('w='+str(w))