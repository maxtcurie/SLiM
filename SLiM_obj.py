import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
import imageio


from interp import interp
from finite_differences import fd_d1_o4

from max_pedestal_finder import find_pedestal_from_data
from max_pedestal_finder import find_pedestal
from read_profiles import read_profile_file
from read_profiles import read_geom_file
from read_EFIT_file import get_geom_pars
from max_stat_tool import Poly_fit
from max_stat_tool import coeff_to_Poly

#Created by Max T. Curie  09/17/2021
#Last edited by Max Curie 09/17/2021
#Supported by scripts in IFS


class mode_finder:
    profile_type=('pfile','ITERDB')
    geomfile_type=('gfile','GENE_tracor')
    def __init__(self,profile_type,profile_name,geomfile_type,geomfile_name,\
            outputpath,path,suffix='.dat',mref=2.,Impurity_charge=6.,\
            zeff_manual=False, show_plot=False):
        n0=1.
        Z=float(Impurity_charge)
        print("suffix"+str(suffix))
        self.profile_name=profile_name
        self.geomfile_name=geomfile_name
        self.outputpath=outputpath
        self.path=path
        self.q_uncertainty=q_uncertainty
        self.profile_uncertainty=profile_uncertainty
        self.doppler_uncertainty=doppler_uncertainty

        if profile_type not in mode_finder.profile_type:
            raise ValueError(f'{profile_type} is not a valid profile type, need to be pfile or ITERDB')
        else: 
            self.profile_type  = profile_type

        if geomfile_type not in mode_finder.geomfile_type:
            raise ValueError(f'{geomfile_type} is not a valid geomfile_type, need to be gfile or GENE_tracor')
        else: 
            self.geomfile_type  = geomfile_type

        #*************Loading the data******************************************
        if profile_type=="ITERDB":
            rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)
        elif profile_type=="pfile":
            rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)
    
    
        if geomfile_type=="gfile": 
            xgrid, q, R_ref= read_geom_file(geomfile_type,geomfile_name,suffix)
        elif geomfile_type=="GENE_tracor":
            xgrid, q, Lref, R_ref, Bref, x0_from_para = read_geom_file(geomfile_type,geomfile_name,suffix)
    
        if geomfile_type=="GENE_tracor" and profile_type!="profile":
            rhot0_range_min=np.argmin(abs(rhot0-xgrid[0]))
            rhot0_range_max=np.argmin(abs(rhot0-xgrid[-1]))
            rhot0=rhot0[rhot0_range_min:rhot0_range_max]
            rhop0=rhop0[rhot0_range_min:rhot0_range_max]
            te0=te0[rhot0_range_min:rhot0_range_max]
            ti0=ti0[rhot0_range_min:rhot0_range_max]
            ne0=ne0[rhot0_range_min:rhot0_range_max]
            ni0=ni0[rhot0_range_min:rhot0_range_max]
            vrot0=vrot0[rhot0_range_min:rhot0_range_max]
    
    
        uni_rhot = np.linspace(min(rhot0),max(rhot0),len(rhot0)*3)
        

        ni_u = interp(rhot0,ni0,uni_rhot)
        te_u = interp(rhot0,te0,uni_rhot)
        ne_u = interp(rhot0,ne0,uni_rhot)
        nz_u = interp(rhot0,nz0,uni_rhot)
        vrot_u = interp(rhot0,vrot0,uni_rhot)
        q      = interp(xgrid,q,uni_rhot)
        tprime_e = -fd_d1_o4(te_u,uni_rhot)/te_u
        nprime_e = -fd_d1_o4(ne_u,uni_rhot)/ne_u
        qprime = fd_d1_o4(q,uni_rhot)/q
        
        print(str(x0_min)+', '+str(x0_max))

        midped, topped=find_pedestal(file_name=geomfile_name, path_name='', plot=False)
        x0_center=midped
        print('mid pedestal is at r/a = '+str(x0_center))
    
        if geomfile_type=="gfile": 
            Lref, Bref, R_major, q0, shat0=get_geom_pars(geomfile_name,x0_center)
    
        print("Lref="+str(Lref))
    
        index_begin=np.argmin(abs(uni_rhot-x0_min))
        index_end  =np.argmin(abs(uni_rhot-x0_max))

        te_u = te_u[index_begin:index_end]
        ne_u = ne_u[index_begin:index_end]
        ni_u = ni_u[index_begin:index_end]
        nz_u = nz_u[index_begin:index_end]
        vrot_u = vrot_u[index_begin:index_end]
        q      = q[index_begin:index_end]
        tprime_e = tprime_e[index_begin:index_end]
        nprime_e = nprime_e[index_begin:index_end]
        qprime   = qprime[index_begin:index_end]
        uni_rhot = uni_rhot[index_begin:index_end]

        center_index = np.argmin(abs(uni_rhot-x0_center))
        
        
        #*************End of loading the data******************************************
    
        #****************Start setting up ******************
        ne=ne_u/(10.**19.)      # in 10^19 /m^3
        ni=ni_u/(10.**19.)      # in 10^19 /m^3
        nz=nz_u/(10.**19.)      # in 10^19 /m^3
        te=te_u/1000.           #in keV
        m_SI = mref *1.6726*10**(-27)
        me_SI = 9.11*10**(-31)
        c  = 1.
        qref = 1.6*10**(-19)
        #refes to GENE manual
        coll_c=2.3031*10**(-5)*Lref*ne/(te)**2*(24-np.log(np.sqrt(ne*10**13)/(te*1000)))
        coll_ei=4.*coll_c*np.sqrt(te*1000.*qref/me_SI)/Lref
        nuei=coll_ei
        beta=403.*10**(-5)*ne*te/Bref**2.
    
        nref=ne_u[center_index]
        te_mid=te_u[center_index]
        Tref=te_u[center_index] * qref
        
        cref = np.sqrt(Tref / m_SI)
        Omegaref = qref * Bref / m_SI / c
        rhoref = cref / Omegaref 
        rhoref_temp = rhoref * np.sqrt(te_u/te_mid) 
        kymin=n0*q0*rhoref/(Lref*x0_center)
        kyGENE =kymin * (q/q0) * np.sqrt(te_u/te_mid) * (x0_center/uni_rhot) #Add the effect of the q varying
        #from mtm_doppler
        omMTM = kyGENE*(tprime_e+nprime_e)
        gyroFreq = 9.79E3/np.sqrt(mref)*np.sqrt(te_u)/Lref
        mtmFreq = omMTM*gyroFreq/(2.*np.pi*1000.)
        omegaDoppler = abs(vrot_u*n0/(2.*np.pi*1000.))
        omega=mtmFreq + omegaDoppler
    
        zeff = ( (ni+Z**2*nz)/ne )[center_index]
        if zeff_manual!=False:
            zeff=zeff_manual
        print('********zeff*********')
        print('zeff='+str(zeff))
        print('********zeff*********')
    
        omega_n_GENE=kyGENE*(nprime_e)       #in cs/a
        omega_n=omega_n_GENE*gyroFreq/(2.*np.pi*1000.)  #in kHz
    
        coll_ei=coll_ei/(2.*np.pi*1000.)  #in kHz

        Lt=1./tprime_e
        Ln=1./nprime_e
        Lq=1./(Lref/(R_ref*q)*qprime)
        
        shat=Ln/Lq
        eta=Ln/Lt
        ky=kyGENE*np.sqrt(2.)
        nu=(coll_ei)/(np.max(omega_n))


        self.rho_s=rhoref
        self.Lref=Lref
        self.x=uni_rhot
        self.shat=shat
        self.eta=eta
        self.ky=ky
        self.nu=nu
        self.zeff=zeff
        self.beta=beta
        self.q=q
        self.q_nominal=q
        self.ome=mtmFreq
        self.Doppler=omegaDoppler

    def q_fit(self,order=5,show=False):
        dq= np.gradient(self.q,self.x)
        x_dq_min=self.x[np.argmin(abs(dq))]

        coeff=Poly_fit(self.x-x_dq_min, self.q, order, show)
        q_fit_coeff=coeff_to_Poly(self.x, x_dq_min, coeff, show)

        return coeff, q_fit_coeff

    def q_modify(self,q_scale,q_shift):
        self.q=self.q*q_scale+q_shift
        return self.q

    def q_back_to_nominal(self):
        self.q=self.q_nominal
        return self.q

    def ome_peak_range(self,peak_percent=0.01):
        y1=self.ome
        x=self.x
        peak_index=np.argmax(self.ome)

        left_x=x[:peak_index]
        right_x=x[peak_index:]
        left_y1 =y1[:peak_index]
        right_y1=y1[peak_index:]
    
        left_index=np.argmin(abs(left_y1  - y1[peak_index]*(1.-peak_percent)))
        right_index=np.argmin(abs(right_y1  - y1[peak_index]*(1.-peak_percent)))
        
        
        self.x_peak=self.x[peak_index]
        self.x_min=left_x[left_index]
        self.x_max=right_x[right_index]

        return self.x_peak,self.x_min,self.x_max

    def Rational_surface(self,n0):
        q=self.q
        uni_rhot=self.x

        x_list=[]
        m_list=[]
        qmin = np.min(q)
        qmax = np.max(q)
    
        m_min = math.ceil(qmin*n0)
        m_max = math.floor(qmax*n0)
        mnums = np.arange(m_min,m_max+1)

        for m in mnums:
            #print(m)
            q0=float(m)/float(n0)
            index0=np.argmin(abs(q-q0))
            if abs(q[index0]-q0)<0.1:
                x_list.append(uni_rhot[index0])
                m_list.append(m)
    
        return x_list, m_list



    def ome_q_surface_demo_plot(self,peak_percent,n_min,n_max,f_min,f_max):
        n_list=np.array(np.arange(n_min,n_max+1),dtype=int)
        x_peak,x_min,x_max=self.ome_peak_range(peak_percent)
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        par1 = host.twinx()
        p1, = par1.plot(self.x, self.q, "b-", label='Safety Factor')
        p2, = par1.plot([x_min]*1000,np.arange(0,20,(20./1000)), color="orange", label='Top '+str(int(peak_percent*100.))+r'$\%$ of drive')
        par1.axvline(x_max, color="orange")
        p3, = host.plot(np.arange(0,1.2,(1.2/1000)),[f_min]*1000, alpha=0.4,color="purple", label='Frequency boundary')
        host.axhline(f_max, color="purple",alpha=0.4)
        x_fill=[x_min,x_min,x_max,x_max]
        y_fill=[f_min,f_max,f_max,f_min]
        #matplotlib.patches.Rectangle((0,total_trans-trans_error),10,2.*trans_error,alpha=0.4)
        host.fill(x_fill,y_fill,color='orange',alpha=0.3)
        

        for n in n_list:
            x_list, m_list =self.Rational_surface(n)
            Unstabel_surface_counter=0

            for j in range(len(m_list)):
                x0=x_list[j]
                m=m_list[j]
                    
                if x_min<x0 and x0<x_max:
                    par1.axvline(x0,color='red')
                    host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                    Unstabel_surface_counter=Unstabel_surface_counter+1
                else:
                    par1.axvline(x0,color='green',alpha=0.3)
    
                    
            if Unstabel_surface_counter>0:
                p4, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
            else:
                p5, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "g-", label=r'Stable $\omega_{*e}$')

        

        host.set_xlim(np.min(self.x),np.max(self.x))
        host.set_ylim(0, f_max*1.2)
        par1.set_ylim(np.min(self.q)*0.8,np.max(self.q)*1.2)
        
        host.set_xlabel(r"$\rho_{tor}$")
        host.set_ylabel(r"$\omega_{*e}$(kHz)")
        par1.set_ylabel("Safety factor")

        
        host.yaxis.label.set_color('black')
        par1.yaxis.label.set_color('black')
        
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors='black', **tkw)
        par1.tick_params(axis='y', colors='black', **tkw)
        #par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [p1, p2, p3, p4, p5]
        
        host.legend(lines, [l.get_label() for l in lines])
        plt.show()


    #this attribute only plot the nearest rational surface for the given toroidal mode number
    def ome_q_surface_demo_plot_clean(self,peak_percent,n_min,n_max,f_min,f_max):
        n_list=np.array(np.arange(n_min,n_max+1),dtype=int)
        x_peak,x_min,x_max=self.ome_peak_range(peak_percent)
        x_peak_plot,x_min_plot,x_max_plot=self.ome_peak_range(peak_percent*10.)
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        #par1 = host.twinx()
        #p1, = par1.plot(self.x, self.q, "b-", label='Safety Factor')
        
        x_fill=[x_min,x_min,x_max,x_max]
        y_fill=[f_min,f_max,f_max,f_min]
        #matplotlib.patches.Rectangle((0,total_trans-trans_error),10,2.*trans_error,alpha=0.4)
        p1, = host.fill(x_fill,y_fill,color='blue',alpha=0.3,label='Unstable area')
        
        
        for n in n_list:
            x_list, m_list =self.Rational_surface(n)
            Unstabel_surface_counter=0

            index=np.argmin(abs(x_list-x_peak))
            x_list=[x_list[index]]
            m_list=[m_list[index]]
            for j in range(len(m_list)):
                x0=x_list[j]
                m=m_list[j]
                if x_min_plot<x0 and x0<x_max_plot:
                    if x_min<x0 and x0<x_max:
                        host.axvline(x0,color='orange',alpha=1)
                        host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                        Unstabel_surface_counter=Unstabel_surface_counter+1
                    else:
                        host.axvline(x0,color='orange',alpha=0.3)
                        host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
            
                    
            if Unstabel_surface_counter>0:
                p2, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
            else:
                p3, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "k-", label=r'Stable $\omega_{*e}$')

        

        host.set_xlim(np.min(self.x),np.max(self.x))
        host.set_ylim(0, np.max(f_max)*1.2)
        #par1.set_ylim(np.min(self.q)*0.8,np.max(self.q)*1.2)
        
        host.set_xlabel(r"$\rho_{tor}$")
        host.set_ylabel(r"$\omega_{*e}$(kHz)")
        #par1.set_ylabel("Safety factor")

        
        host.yaxis.label.set_color('black')
        #par1.yaxis.label.set_color('black')
        
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors='black', **tkw)
        #par1.tick_params(axis='y', colors='black', **tkw)
        #par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [p1,p2,p3]
        
        #host.legend(lines, ['Unstable area',r'Unstable $\omega_{*e}$',r'Stable $\omega_{*e}$'])
        host.legend(lines, [l.get_label() for l in lines])
        plt.show()

    
    def ome_q_surface_demo_plot_frequency_band(self,peak_percent,n_min,n_max,f_min_list,f_max_list):
        n_list=np.array(np.arange(n_min,n_max+1),dtype=int)
        x_peak,x_min,x_max=self.ome_peak_range(peak_percent)
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)

        par1 = host.twinx()
        p1, = par1.plot(self.x, self.q, "b-", label='Safety Factor')
        p2, = par1.plot([x_min]*1000,np.arange(0,20,(20./1000)), color="orange", label='Top '+str(int(peak_percent*100.))+r'$\%$ of drive')
        par1.axvline(x_max, color="orange")
        #p3, = host.plot(np.arange(0,1.2,(1.2/1000)),[f_min]*1000, alpha=0.4,color='purple', label='Frequency bands')
        #host.axhline(f_max, color="purple",alpha=0.4)
        

        for n in n_list:
            x_list, m_list =self.Rational_surface(n)
            Unstabel_surface_counter=0

            for j in range(len(m_list)):
                x0=x_list[j]
                m=m_list[j]
                    
                if x_min<x0 and x0<x_max:
                    par1.axvline(x0,color='red')
                    host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                    Unstabel_surface_counter=Unstabel_surface_counter+1
                else:
                    par1.axvline(x0,color='green',alpha=0.3)
    
                    
            if Unstabel_surface_counter>0:
                p4, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
            else:
                p5, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "g-", label=r'Stable $\omega_{*e}$')

        for i in range(len(f_max_list)):
            x_min,x_max=0,1
            f_min=f_min_list[i]
            f_max=f_max_list[i]
            x_fill=[x_min,x_min,x_max,x_max]
            y_fill=[f_min,f_max,f_max,f_min]
            #matplotlib.patches.Rectangle((0,total_trans-trans_error),10,2.*trans_error,alpha=0.4)
            host.fill(x_fill,y_fill,color='orange',alpha=0.3)
        
        

        host.set_xlim(np.min(self.x),np.max(self.x))
        host.set_ylim(0, np.max(n_list)*np.max(self.ome+self.Doppler)*1.2)
        par1.set_ylim(np.min(self.q)*0.8,np.max(self.q)*1.2)
        
        host.set_xlabel(r"$\rho_{tor}$")
        host.set_ylabel(r"$\omega_{*e}$(kHz)")
        par1.set_ylabel("Safety factor")

        
        host.yaxis.label.set_color('red')
        par1.yaxis.label.set_color('blue')
        
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors='red', **tkw)
        par1.tick_params(axis='y', colors='blue', **tkw)
        #par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [p1, p2, p4, p5]
        
        host.legend(lines, [l.get_label() for l in lines])
        plt.show()


    def x_sigma(self,peak_percent=0.01):
        a=2
        

