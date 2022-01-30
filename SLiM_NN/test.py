import numpy as np
if 1==1:
    nu_list=np.arange(0.1,10.,0.5)
    zeff_list=np.arange(1,2.5,0.2)
    eta_list=np.arange(0.5,3.,0.2)
    shat_list=np.arange(0.02,0.1,0.01)
    beta_list=np.arange(0.0005,0.003,0.0003)
    ky_list=np.arange(0.01,0.1,0.01)
    mu_list=np.arange(0,4.,0.1)
    xstar=10.
    ModIndex=1 #global dispersion

    para_list=[]
    for nu in nu_list:
        for zeff in zeff_list:
            for eta in eta_list:
                for shat in shat_list:
                    for beta in beta_list:
                        for ky in ky_list:
                            for mu in mu_list:
                                para_list.append([nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar])
    print(len(para_list))

else:
    print(1)