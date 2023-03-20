from SLiM_phys import Dispersion_func

Run_mode=2      # mode1: fast mode
                # mode2: slow mode(global)
                # mode3: slow mode(local) 
                # mode4: slow mode manual(global)
                # mode5: slow slow mode(global)
                # mode6: NN mode (global)

[nu,  zeff, eta, shat,  beta,  ky,   mu, xstar]=\
[2.5, 2.5,  1.5, 0.001, 0.001, 0.15, 0., 10.  ]

w0=Dispersion_func.Dispersion(nu,zeff,eta,shat,beta,ky,mu,xstar,Run_mode=Run_mode)

print('w0='+str(w0))
