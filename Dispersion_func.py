
def Dispersion(nu,zeff,eta,shat,beta,ky,mu,xstar,Run_mode=1):
    
    if Run_mode==1: #fast mode
        w0=1.+eta   #frequency = ome = 1+eta
    if Run_mode==2: #slow mode (global)
        w0=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,1,abs(mu),xstar)
    if Run_mode==3: #slow mode (local)
        w0=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,0,abs(mu),xstar)
    elif Run_mode==4: #manual mode
        w0=VectorFinder_manual(nu,zeff,eta,shat,beta,ky,1,abs(mu),xstar)
    elif Run_mode==5: #slow slow mode (global)
        w0=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,1,abs(mu),xstar)
    elif Run_mode==6: #NN mode
        from .SLiM_NN.Dispersion_NN import Dispersion_NN
        Dispersion_NN_obj=Dispersion_NN(path)
        w0=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,abs(mu),xstar)

    return w0



def Dispersion_list(para_list,Run_mode=1):
    
    if Run_mode==6: #NN mode
        from .SLiM_NN.Dispersion_NN import Dispersion_NN
        Dispersion_NN_obj=Dispersion_NN(path)
        w0_list=[]
        for [nu,zeff,eta,shat,beta,ky,mu,xstar] in para_list:
            w0=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar)
            w0_list.append(w0)
    else:
        for [nu,zeff,eta,shat,beta,ky,mu,xstar] in para_list:
            w0=Dispersion(nu,zeff,eta,shat,beta,ky,mu,xstar,Run_mode)
            w0_list.append(w0)

    return w0_list

