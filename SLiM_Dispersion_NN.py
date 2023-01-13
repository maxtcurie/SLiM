from SLiM_NN.Dispersion_NN import Dispersion_NN

path='./SLiM_NN/Trained_model/'

[nu, zeff, eta, shat,  beta,  ky,   mu, xstar]=\
[1., 1.5,  1.5, 0.001, 0.001, 0.03, 0., 10.  ]

Dispersion_NN_obj=Dispersion_NN(path)
w=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar)

print('w='+str(w))