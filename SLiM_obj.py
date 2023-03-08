import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import optimize
import math
import csv
import imageio
from tqdm import tqdm
import sys as sys
sys.path.insert(1, './Tools')

from SLiM_NN.Dispersion_NN import Dispersion_NN
from interp import interp
from finite_differences import fd_d1_o4
from Gaussian_fit import gaussian_fit
from Gaussian_fit import gaussian_fit_GUI
from max_pedestal_finder import find_pedestal_from_data
from max_pedestal_finder import find_pedestal
from read_profiles import read_profile_file
from read_profiles import read_geom_file
from read_EFIT_file import get_geom_pars
from write_iterdb import output_iterdb
from max_stat_tool import Poly_fit
from max_stat_tool import coeff_to_Poly
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_manual
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto_Extensive
#Created by Max T. Curie  09/17/2021
#Last edited by Max Curie 09/17/2021
#Supported by scripts in IFS


class mode_finder:
    profile_type=('pfile','ITERDB')
    geomfile_type=('gfile','GENE_tracor')
    
    def __init__(self,profile_type,profile_name,geomfile_type,geomfile_name,\
            outputpath,inputpath,x0_min,x0_max,zeff_manual=-1,suffix='.dat',\
            mref=2.,Impurity_charge=6.):
        n0=1.
        Z=float(Impurity_charge)
        #print("suffix"+str(suffix))
        self.profile_name=profile_name
        self.geomfile_name=geomfile_name
        self.outputpath=outputpath
        self.inputpath=inputpath
        self.x0_min=x0_min
        self.x0_max=x0_max

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
    
        
        uni_rhot = np.linspace(min(rhot0),max(rhot0),2000)
        
        uni_rhop = interp(rhot0,rhop0,uni_rhot)
        ti_u = interp(rhot0,ti0,uni_rhot)
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

        ti_u = ti_u[index_begin:index_end]
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
        te=te_u/1000.           # in keV
        ti=ti_u/1000.           # in keV
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
        omegaDoppler = vrot_u*n0/(2.*np.pi*1000.)
        omega=mtmFreq + omegaDoppler
        

        if zeff_manual==-1:
            zeff = (  ( ni+nz*(float(Z)**2.) )/ne  )[center_index]
        else:
            zeff=zeff_manual
        print('********zeff*********')
        print('zeff='+str(zeff))
        print('********zeff*********')
        
        if 1==0:
            plt.clf()
            plt.plot(uni_rhot,omegaDoppler)
            plt.plot(uni_rhot,mtmFreq)
            plt.plot(uni_rhot,omega)
            plt.grid()
            plt.show()

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

        self.mref=mref
        self.Z=Z
        self.Bref=Bref

        self.Lt=Lt
        self.Ln=Ln
        self.Lq=Lq

        self.ne=ne      # in 10^19 /m^3
        self.ni=ni      # in 10^19 /m^3
        self.nz=nz      # in 10^19 /m^3
        self.te=te      # in keV 
        self.ti=ti      # in keV     
        self.nref=nref
        self.r_sigma=(1./shat)*( (me_SI/m_SI)**0.5 )
        self.R_ref=R_ref
        self.cs_to_kHz=gyroFreq/(2.*np.pi*1000.)
        self.omn=omega_n    #omega_n in kHz
        self.cs=cref
        self.rho_s=rhoref*np.sqrt(2.)
        self.Lref=Lref
        self.x=uni_rhot
        self.rhop=uni_rhop
        self.shat=shat
        self.eta=eta
        self.ky=ky
        self.nu=nu
        self.zeff=zeff
        self.beta=beta
        self.q=q
        self.ome=mtmFreq
        self.Doppler=omegaDoppler
        self.coll_ei=coll_ei
        self.gyroFreq=gyroFreq

        self.x_nominal=uni_rhot
        self.nref_nominal=nref
        self.ne_nominal=self.ne     
        self.te_nominal=self.te       
        self.r_sigma_nominal=self.r_sigma
        self.cs_to_kHz_nominal=self.cs_to_kHz
        self.omn_nominal=self.omn
        self.cs_nominal=self.cs
        self.rho_s_nominal=self.rho_s
        self.shat_nominal=self.shat
        self.eta_nominal=self.eta
        self.ky_nominal=self.ky
        self.nu_nominal=self.nu
        self.zeff_nominal=self.zeff
        self.beta_nominal=self.beta
        self.q_nominal=self.q
        self.ome_nominal=self.ome
        self.Doppler_nominal=self.Doppler
        self.coll_ei_nominal=self.coll_ei
        self.gyroFreq_nominal=self.gyroFreq

        self.q_scale=1.
        self.q_shift=0.
        self.shat_scale=1.
        self.ne_scale=1.
        self.ne_shift=0.
        self.te_scale=1.
        self.Doppler_scale=1.


    def __str__(self):
        try:
            self.x_peak
        except:
            self.ome_peak_range(0.1)
        index=np.argmin(abs(self.x-self.x_peak))
        
        return f'Parameter at peak:\n\
        shat={self.shat[index]},\n\
        eta={self.eta[index]},\n\
        ky={self.ky[index]},\n\
        nu={self.nu[index]},\n\
        zeff={self.zeff},\n\
        beta={self.beta[index]},\n\
        r_sigma={self.r_sigma[index]},\n\
        xstar={self.xstar}'


    def set_xstar(self,xstar):
        self.xstar=xstar
        #setter for xstar

    
    def find_pedestal(self,x,p):
        dp=np.gradient(p,x)  #Second order of pressure
        ddp   = np.gradient(dp,x)  #Second order of pressure
        midped = x[np.argmin(dp)]
        topped = x[np.argmin(ddp)]
        return topped,midped

    def q_scale_for_alignment(self,q_at_peak):
        x=self.x
        q=self.q
        ome=self.ome 

        

        index_tmp_=np.argmax(ome)
        q_scale_=q_at_peak/q[index_tmp_]

        if 1==0:
            print(q_at_peak)
            plt.clf()
            plt.plot(x,q)
            plt.plot(x,q*q_scale_)
            plt.axvline(x[index_tmp_])
            plt.show()

        #x2=np.linspace(np.min(x),np.max(x),0.01*abs(x[1]-x[0]))
        #q2=np.interp(x2,x,q)
        #ome2=np.interp(x2,x,ome)

        #index_tmp_=np.argmax(ome2)
        #q_scale_=q_at_peak/q2[index_tmp_]

        return q_scale_

    def inside_freq_band_check(self,f,freq_min_list,freq_max_list):
        for freq_min, freq_max in zip(freq_min_list,freq_max_list):
            if freq_min<=f and f<=freq_max:
                return True
                break


    def profile_fit(self,show_plot=False):
        x=self.x
        ne=self.ne
        te=self.te

        if show_plot==True:
            fig, ax=plt.subplots(nrows=3,ncols=1,sharex=True) 
            ax[0].plot(x,ne,label='original')
            ax[0].set_ylabel(r'$n_e(10^19/m^3)$',fontsize=20)
            ax[1].plot(x,te,label='original')
            ax[1].set_ylabel(r'$T_e(keV)$',fontsize=20)
            ax[1].set_xlabel(r'$\rho_{tor}$',fontsize=20)
            plt.tight_layout()
            plt.show()

        ne_popt, pcov = optimize.curve_fit(tanh_func, x, ne)
        ne_mid_ped, ne_ped_width, ne_height,ne_edge=ne_popt[0], ne_popt[1], ne_popt[2], ne_popt[3]

        te_popt, pcov = optimize.curve_fit(tanh_func, x, te)
        te_mid_ped, te_ped_width, te_height,te_edge=te_popt[0], te_popt[1], te_popt[2], te_popt[3]

        if show_plot==True:
            ne_fit=tanh_func(x, ne_popt[0], ne_popt[1], ne_popt[2], ne_popt[3])
            te_fit=tanh_func(x, te_popt[0], te_popt[1], te_popt[2], te_popt[3])

            fig, ax=plt.subplots(nrows=2,ncols=1,sharex=True) 
            ax[0].plot(x,ne,label='original')
            ax[0].plot(x,ne_fit,label='fit')
            ax[0].legend()
            ax[0].set_ylabel(r'$n_e(10^19/m^3)$',fontsize=10)
            ax[1].plot(x,te,label='original')
            ax[1].plot(x,te_fit,label='fit')
            ax[1].set_ylabel(r'$T_e(keV)$',fontsize=10)
            ax[1].set_xlabel(r'$\rho_{tor}$',fontsize=10)
            plt.tight_layout()
            plt.show()
        
        self.ne_mid_ped=ne_mid_ped
        self.ne_ped_width=ne_ped_width
        self.ne_height=ne_height
        self.ne_edge=ne_edge
        self.te_mid_ped=te_mid_ped
        self.te_ped_width=te_ped_width
        self.te_height=te_height
        self.te_edge=te_edge

        return ne_mid_ped, ne_ped_width, ne_height,ne_edge,\
                te_mid_ped, te_ped_width, te_height,te_edge


    def q_fit(self,q_order=5,show_plot=False):
        dq= np.gradient(self.q,self.x)
        x_dq_min=x[np.argmin(abs(dq))]
        q_coeff=Poly_fit(self.x-x_dq_min, self.q, q_order, show=False)
        q_fit=coeff_to_Poly(self.x, x_dq_min, q_coeff, show=False)
        if show_plot==Ture:
            plt.clf()
            plt.plot(self.x,self.q,label='orginal')
            plt.plot(self.x,q_fit,label='fit')
            plt.legend()
            plt.show()

            
    def modify_profile(self,\
                    q_scale=1.,q_shift=0.,\
                    shat_scale=1.,\
                    ne_scale=1.,te_scale=1.,\
                    ne_shift=0.,te_shift=0.,\
                    Doppler_scale=1.,\
                    show_plot=False):
        
        #************start of modification***********
        #********************************************

        self.q_scale=q_scale
        self.q_shift=q_shift
        self.shat_scale=shat_scale
        self.ne_scale=ne_scale
        self.ne_shift=ne_shift
        self.te_scale=te_scale
        self.Doppler_scale=Doppler_scale

        #**********************************
        #*************modify ne************
        try:
            self.ne_weight
        except:
            ne_top_ped,ne_mid_ped=self.find_pedestal(self.x,self.ne)
            self.ne_mid_ped=ne_mid_ped
            ne_width=ne_mid_ped-ne_top_ped

            self.ne_weight = 0.5+np.tanh((self.x-ne_top_ped)/ne_width)/2

        index=np.argmin(abs(self.x-self.ne_mid_ped))
        ne_mod=self.ne*(1.+(ne_scale-1.)*self.ne_weight)+(ne_shift)*self.ne[index]
        #*************modify ne************
        #**********************************

        #**********************************
        #*************modify Te************
        try:
            self.te_weight
        except:
            te_top_ped,te_mid_ped=self.find_pedestal(self.x,self.te)
            self.te_mid_ped=te_mid_ped
            te_width=te_mid_ped-te_top_ped
            self.te_weight = 0.5+np.tanh((self.x-te_top_ped)/te_width)/2

        if 1==0:
            print('te_top_ped='+str(te_top_ped))
            print('te_mid_ped='+str(te_mid_ped))
            print('te_width='+str(te_width))

        index=np.argmin(abs(self.x-self.te_mid_ped))

        te_mod=self.te*(1.+(te_scale-1.)*self.te_weight)+(te_shift)*self.te[index]
        #*************modify Te************
        #**********************************

        mid_ped,top_ped=self.find_pedestal(self.x,self.ome)


        width=(mid_ped-top_ped)*10.
        try:
            self.shat_weight
        except:
            self.shat_weight = -1.*np.tanh((self.x-mid_ped)/width)/2
        
        q_mod=self.q*q_scale*(1.+(shat_scale-1.)*self.shat_weight)+q_shift

        Doppler_mod=self.Doppler*Doppler_scale

        #show_plot=True

        if show_plot==True:
            fig, ax=plt.subplots(nrows=4,ncols=1,sharex=True) 
            ax[0].plot(self.x,self.ne,label='original')
            ax[0].plot(self.x,ne_mod,label='modified')
            ax[0].legend()
            ax[0].set_ylabel(r'$n_e(10^19/m^3)$',fontsize=10)
            ax[1].plot(self.x,self.te)
            ax[1].plot(self.x,te_mod)
            ax[1].set_ylabel(r'$T_e(keV)$',fontsize=10)
            ax[2].plot(self.x,self.q)
            ax[2].plot(self.x,q_mod)
            ax[2].set_ylabel(r'$q$',fontsize=10)
            ax[3].plot(self.x,self.Doppler)
            ax[3].plot(self.x,Doppler_mod)
            ax[3].set_ylabel(r'$f_{Doppler}(kHz)$',fontsize=10)
            ax[3].set_xlabel(r'$\rho_{tor}$',fontsize=10)
            plt.tight_layout()
            plt.show()

        #*************end of modification************
        #********************************************


        tprime_e = -fd_d1_o4(te_mod,self.x)/te_mod
        nprime_e = -fd_d1_o4(ne_mod,self.x)/ne_mod
        qprime = fd_d1_o4(q_mod,self.x)/q_mod

        center_index=np.argmin(abs(self.x-mid_ped))

        Tref_mod=te_mod[center_index]

        
        #*********start of calculation******
        #****************Start setting up ******************
        #get the data from attribute
        ni=self.ni
        nz=self.nz

        te_u=te_mod*1000.
        ne_u=ne_mod*10.**19.
        ni_u=ni*10.**19.
        nz_u=nz*10.**19.
        mref=self.mref
        Lref=self.Lref
        Bref=self.Bref
        uni_rhot=self.x
        Z=self.Z
        R_ref=self.R_ref


        x0_center=mid_ped
        ne=ne_mod
        te=te_mod
        q=q_mod

        
        n0=1.
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
        Tref=te_u[center_index]*qref
        q0=q_mod[center_index]
        
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
        omegaDoppler = Doppler_mod
        omega=mtmFreq + omegaDoppler
        
        zeff=self.zeff*nref/self.nref

    
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

        self.x=self.x[2:-2] 
        self.nref=nref
        self.ne=ne[2:-2]      # in 10^19 /m^3
        self.ni=ni[2:-2]      # in 10^19 /m^3
        self.nz=nz[2:-2]      # in 10^19 /m^3
        self.te=te[2:-2]      # in keV      
        self.r_sigma=(1./shat[:-2])*( (me_SI/m_SI)**0.5 )
        self.cs_to_kHz=gyroFreq[:-2]/(2.*np.pi*1000.)
        self.omn=omega_n[2:-2]    #omega_n in kHz
        self.cs=cref
        self.rho_s=rhoref*np.sqrt(2.)
        self.shat=shat[2:-2]
        self.eta=eta[2:-2]
        self.ky=ky[2:-2]
        self.nu=nu[2:-2]
        self.zeff=zeff
        self.beta=beta[2:-2]
        self.q=q[2:-2]
        self.ome=mtmFreq[2:-2]
        self.Doppler=omegaDoppler[2:-2]
        self.coll_ei=coll_ei[2:-2]
        self.gyroFreq=gyroFreq[2:-2]
        
        self.ome_peak_range(0.1)
        #**********end of calculation*******
        if show_plot==True:
            fig, ax=plt.subplots(nrows=7,\
            ncols=1,sharex=True) 
        
            ax[0].plot(self.x_nominal,self.shat_nominal,label='nominal')
            ax[0].plot(self.x,self.shat,label='modified')
            ax[0].legend()
            ax[0].set_ylabel(r'$L_{ne}/L_{q}$')
            ax[1].plot(self.x_nominal,self.eta_nominal)
            ax[1].plot(self.x,self.eta)
            ax[1].set_ylabel(r'$L_{ne}/L_{Te}$')
            ax[2].plot(self.x_nominal,self.ky_nominal)
            ax[2].plot(self.x,self.ky)
            ax[2].set_ylabel(r'$k_y\rho_s$')
            ax[3].plot(self.x_nominal,self.nu_nominal)
            ax[3].plot(self.x,self.nu)
            ax[3].set_ylabel(r'$\nu_{ei}/\omega_{*ne}$')
            ax[4].plot(self.x_nominal,self.beta_nominal)
            ax[4].plot(self.x,self.beta)
            ax[4].set_ylabel(r'$\beta$')
            ax[5].plot(self.x_nominal,self.q_nominal)
            ax[5].plot(self.x,self.q)
            ax[5].set_ylabel(r'$q$')
            ax[6].plot(self.x_nominal,self.ome_nominal)
            ax[6].plot(self.x,self.ome)
            ax[6].set_ylabel(r'$\omega_{*e}(kHz)$')
            ax[6].set_xlabel(r'$\rho_{tor}$')  
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()  

    
    def reset_profile(self):
        self.x=self.x_nominal
        self.ne=self.ne_nominal   
        self.te=self.te_nominal
        self.r_sigma=self.r_sigma_nominal
        self.cs_to_kHz=self.cs_to_kHz_nominal
        self.omn=self.omn_nominal
        self.cs=self.cs_nominal
        self.rho_s=self.rho_s_nominal
        self.shat=self.shat_nominal
        self.eta=self.eta_nominal
        self.ky=self.ky_nominal
        self.nu=self.nu_nominal
        self.zeff=self.zeff_nominal
        self.beta=self.beta_nominal
        self.q=self.q_nominal
        self.ome=self.ome_nominal
        self.Doppler=self.Doppler_nominal
        self.coll_ei=self.coll_ei_nominal
        self.gyroFreq=self.gyroFreq_nominal
        self.nref=self.nref_nominal

        self.q_scale=1.
        self.q_shift=0.
        self.shat_scale=1.
        self.ne_scale=1.
        self.ne_shift=0.
        self.te_scale=1.
        self.Doppler_scale=1.


    def q_modify(self,q_scale,q_shift,shat_scale=1.):
        self.q_scale=q_scale
        self.q_shift=q_shift
        self.shat_scale=shat_scale

        mid_ped,top_ped=self.find_pedestal(self.x,self.te)

        width=(mid_ped-top_ped)*10.
        try:
            self.shat_weight
        except:
            self.shat_weight = -1.*np.tanh((self.x-mid_ped)/width)/2

        q_mod=self.q*q_scale*(1.+(shat_scale-1.)*self.shat_weight)+q_shift

        self.ome= self.ome*q_mod/self.q
        self.omn= self.omn*q_mod/self.q
        self.shat= self.shat*( (q_scale*self.q)/q_mod )
        self.ky=self.ky*q_mod/self.q

        self.q=q_mod

        return self.q


    def q_back_to_nominal(self):
        self.q=self.q_nominal
        self.ome= self.ome_nominal
        self.omn= self.omn_nominal
        self.shat= self.shat_nominal
        self.ky=self.ky_nominal

        self.q_scale=1.
        self.q_shift=0.
        self.shat_scale=1.


    def omega_gaussian_fit_GUI(self,root,x,data,rhoref,Lref):
        amplitude,mean,stddev=gaussian_fit_GUI(root,x,data)
        #print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
        mean_rho=mean*Lref/rhoref         #normalized to rhoi
        xstar=abs(stddev*Lref/rhoref)
        
        popt=[0]*3
        popt[0] = amplitude
        popt[1] = mean     
        popt[2] = stddev   
    
        #print(popt)
        #print(mean_rho,xstar)
    
        return mean_rho,xstar
  

    def omega_gaussian_fit(self,manual=False,fit_type=0):
        x=self.x 
        data=self.ome 
        rhoref=self.rho_s 
        Lref=self.Lref 

        amplitude,mean,stddev=gaussian_fit(x,data,manual,fit_type)
        #print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
        mean_rho=mean*Lref/rhoref         #normalized to rhoi
        xstar=abs(stddev*Lref/rhoref)
        
        popt=[0]*3
        popt[0] = amplitude
        popt[1] = mean     
        popt[2] = stddev   

        #print(popt)
        #print(mean_rho,xstar)
    
        return mean_rho,xstar
  

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
        
        q_prime=(fd_d1_o4(q,uni_rhot)/q)[1:-1]
        peak_index=np.argmin(abs(q_prime))
        
        qmin = np.min(q)
        qmax = np.max(q)
        m_min = math.ceil(qmin*n0)
        m_max = math.floor(qmax*n0)
        mnums = np.arange(m_min,m_max+1)
        dq=np.max(abs(q[:-1]-q[1:]))
        
        if np.min(q_prime)<-0.001: #for non-monotonic increasing q profiles
            for m in mnums:
                #print(m)
                x_list_temp=[]
                index_list_temp=[]
                q0=float(m)/float(n0)
                for index0 in range(len(q)):
                    if abs(q[index0]-q0)<dq:
                        x_list_temp.append(uni_rhot[index0])
                        index_list_temp.append(index0)
                index_diff=np.array(index_list_temp[:-1],dtype=int)-np.array(index_list_temp[1:],dtype=int)
                #print(index_diff)
                #print(x_list_temp)
                #print(index_list_temp)
                if len(x_list_temp)>0:
                    x_list.append(x_list_temp[0])
                    m_list.append(m)
                for i in range(len(index_diff)):
                    if index_diff[i]==-1:
                        pass 
                    else:
                        x_list.append(x_list_temp[i])
                        m_list.append(m)
    
            #print(x_list)
            #print(m_list)
            return x_list, m_list
        else: #for monotonic increasing q profiles
            for m in mnums:
                q0=float(m)/float(n0)
                index0=np.argmin(abs(q-q0))
                if abs(q[index0]-q0)<0.1:
                    x_list.append(uni_rhot[index0])
                    m_list.append(m)
                
            #print(x_list)
            #print(m_list)
            return x_list, m_list


    def Rational_surface_peak_surface(self,n0):
        try:
            x_peak=self.x_peak
        except:
            x_peak,x_min,x_max=self.ome_peak_range(peak_percent=0.1)

        x_list, m_list=self.Rational_surface(n0)
        index=np.argmin(abs(x_list-x_peak))
        x_surface_near_peak=x_list[index]
        m_surface_near_peak=m_list[index]
        
        return x_surface_near_peak, m_surface_near_peak


    def Rational_surface_top_surfaces(self,n0,top=1):
        try:
            x_peak=self.x_peak
        except:
            x_peak,x_min,x_max=self.ome_peak_range(peak_percent=0.1)

        x_list, m_list=self.Rational_surface(n0)
        temp_ndarray=(np.array([x_list, m_list,abs(x_list-x_peak)])).transpose() 
        surface_df = pd.DataFrame(temp_ndarray,\
                    columns = ['x_list','m_list','peak_distance'])
        #print(surface_df)
        surface_df_sort=surface_df.sort_values(by='peak_distance')

        #print(surface_df_sort)
        #input()
        if len(surface_df_sort['x_list'])<top:
            x_surface_near_peak_list=surface_df_sort['x_list']
            m_surface_near_peak_list=surface_df_sort['m_list']
        else:
            x_surface_near_peak_list=surface_df_sort['x_list'][:top]
            m_surface_near_peak_list=surface_df_sort['m_list'][:top]
        x_surface_near_peak_list=np.array(x_surface_near_peak_list)
        m_surface_near_peak_list=np.array(m_surface_near_peak_list)
        return x_surface_near_peak_list, m_surface_near_peak_list


    def parameter_for_dispersion(self,x0,n):
        index=np.argmin(abs(x0-self.x))

        factor_temp=np.sqrt(\
                (float(n)*self.q[index]/self.Lref)**2.\
                +(float(n)/self.R_ref)**2.)\
                *self.Lref/self.q[index]

        self.factor=factor_temp

        nu=self.nu[index]/factor_temp
        zeff=self.zeff
        eta=self.eta[index]
        shat=self.shat[index]
        beta=self.beta[index]
        ky=self.ky[index]*factor_temp

        try:
            x_peak=self.x_peak
        except:
            x_peak,x_min,x_max=self.ome_peak_range(peak_percent=0.1)

        try:
            xstar=self.xstar
        except:
            xstar=0.
        mu=(x0-x_peak)*self.Lref/(self.rho_s)
        return nu,zeff,eta,shat,beta,ky,mu,xstar


    def Dispersion(self,nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar,manual=False):
        if manual==True:
            w0=VectorFinder_manual(nu,zeff,eta,shat,beta,ky,ModIndex,abs(mu),xstar)
        elif manual==5:
            w0=VectorFinder_auto_Extensive(nu,zeff,eta,shat,beta,ky,ModIndex,abs(mu),xstar)
        else:
            w0=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,ModIndex,abs(mu),xstar)
        return w0
 

    def plot_parameter(self):
        fig, ax=plt.subplots(nrows=7,\
            ncols=1,sharex=True) 
        
        ax[0].plot(self.x,self.shat)
        ax[0].set_ylabel(r'$L_{ne}/L_{q}$')
        ax[1].plot(self.x,self.eta)
        ax[1].set_ylabel(r'$L_{ne}/L_{Te}$')
        ax[2].plot(self.x,self.ky)
        ax[2].set_ylabel(r'$k_y\rho_s$')
        ax[3].plot(self.x,self.nu)
        ax[3].set_ylabel(r'$\nu_{ei}/\omega_{*ne}$')
        ax[4].plot(self.x,self.beta)
        ax[4].set_ylabel(r'$\beta$')
        ax[5].plot(self.x,self.q,label=r'$q$')
        ax[5].set_ylabel(r'$q$')
        ax[6].plot(self.x,self.ome)
        ax[6].set_ylabel(r'$\omega_{*e}(kHz)$')

        ax[6].set_xlabel(r'$\rho_{tor}$')  
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()  


    def Plot_q_scale_rational_surfaces_colored(self,peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list):
        marker_size=5
        x_peak,x_stability_boundary_min,x_stability_boundary_max=self.ome_peak_range(peak_percent)

        x_min_index=np.argmin(abs(self.x-x_stability_boundary_min))
        x_max_index=np.argmin(abs(self.x-x_stability_boundary_max))
        
        q_range=self.q[x_min_index:x_max_index]

        x_list=[]
        y_list=[]
        x_list_error=[]
        y_list_error=[]
        
        fig, ax = plt.subplots()

        ax.fill_between(self.x, self.q*(1.-q_uncertainty), self.q*(1.+q_uncertainty), color='blue', alpha=.0)
        ax.axvline(x_stability_boundary_min,color='orange',alpha=0.6)
        ax.axvline(x_stability_boundary_max,color='orange',alpha=0.6)
        ax.plot(self.x,self.q,label=r'Safety factor $q_0$')
        
        
        for i in range(len(n_list)):
            n=n_list[i]
            x1=[]
            y1=[]
            x1_error=[]
            y1_error=[]

            qmin = np.min(self.q*(1.-q_uncertainty))
            qmax = np.max(self.q*(1.+q_uncertainty))
        
            m_min = math.ceil(qmin*n)
            m_max = math.floor(qmax*n)
            m_list=np.arange(m_min,m_max+1)

            for m in m_list:
                surface=float(m)/float(n)
                if np.min(q_range)*(1.-q_uncertainty)<surface and surface<np.max(q_range)*(1.+q_uncertainty):
                    x1.append(0.5*(x_stability_boundary_max+x_stability_boundary_min))
                    y1.append(surface)
                    x1_error.append(0.5*(x_stability_boundary_max-x_stability_boundary_min))
                    y1_error.append(0)
            if color_list==-1:
                ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',ms=marker_size,label='n='+str(n))
            else:
                ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',color=color_list[i],linestyle='none',ms=marker_size,label='n='+str(n))
            
            x_list.append(x1)
            y_list.append(y1)
            x_list_error.append(x1_error)
            y_list_error.append(y1_error)

        ax.plot(self.x,self.q*q_scale+q_shift,color='orange',label=r'Modified $q=$'+str(q_scale)+r'$*q_0$')
        ax.set_xlim(self.x0_min,self.x0_max)
        ax.set_xlabel(r'$\rho_{tor}$')
        ax.set_ylabel('Safety factor')
        plt.legend(loc='upper left')
        plt.show()
 

    #color_lsit=-1 for auto color assignment
    def Plot_q_scale_rational_surfaces_colored_obj(self,peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list):
        marker_size=5
        x_peak,x_stability_boundary_min,x_stability_boundary_max=self.ome_peak_range(peak_percent)

        x_min_index=np.argmin(abs(self.x-x_stability_boundary_min))
        x_max_index=np.argmin(abs(self.x-x_stability_boundary_max))
        
        q_range=self.q[x_min_index:x_max_index]

        x_list=[]
        y_list=[]
        x_list_error=[]
        y_list_error=[]
        fig = Figure(figsize = (5, 5),\
                        dpi = 100)
        plot1 = fig.add_subplot(111)
        
        plot1.fill_between(self.x, self.q*(1.-q_uncertainty), self.q*(1.+q_uncertainty), color='blue', alpha=.0)
        plot1.axvline(x_stability_boundary_min,color='orange',alpha=0.6)
        plot1.axvline(x_stability_boundary_max,color='orange',alpha=0.6)
        plot1.plot(self.x,self.q,label=r'Safety factor $q_0$')
        
        
        for i in range(len(n_list)):
            n=n_list[i]
            x1=[]
            y1=[]
            x1_error=[]
            y1_error=[]

            qmin = np.min(self.q*(1.-q_uncertainty))
            qmax = np.max(self.q*(1.+q_uncertainty))
        
            m_min = math.ceil(qmin*n)
            m_max = math.floor(qmax*n)
            m_list=np.arange(m_min,m_max+1)

            for m in m_list:
                surface=float(m)/float(n)
                if np.min(q_range)*(1.-q_uncertainty)<surface and surface<np.max(q_range)*(1.+q_uncertainty):
                    x1.append(0.5*(x_stability_boundary_max+x_stability_boundary_min))
                    y1.append(surface)
                    x1_error.append(0.5*(x_stability_boundary_max-x_stability_boundary_min))
                    y1_error.append(0)
            if color_list==-1:
                plot1.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',ms=marker_size,label='n='+str(n))
            else:
                plot1.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',color=color_list[i],linestyle='none',ms=marker_size,label='n='+str(n))
            
            x_list.append(x1)
            y_list.append(y1)
            x_list_error.append(x1_error)
            y_list_error.append(y1_error)

        plot1.plot(self.x,self.q*q_scale+q_shift,color='orange',label=r'Modified $q=$'+str(q_scale)+r'$*q_0$')
        plot1.set_xlim(self.x0_min,self.x0_max)
        plot1.set_xlabel(r'$\rho_{tor}$')
        plot1.set_ylabel('Safety factor')
        plot1.legend(loc='upper left')
        

        return fig


    def Plot_q_scale_rational_surfaces_red_and_green(self,peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list):
        marker_size=5
        #calculate the radial stability boundary
        x_peak,x_stability_boundary_min,x_stability_boundary_max=self.ome_peak_range(peak_percent)

        x_min_index=np.argmin(abs(self.x-x_stability_boundary_min))
        x_max_index=np.argmin(abs(self.x-x_stability_boundary_max))

        q_range=self.q[x_min_index:x_max_index]

        x_list=[]
        y_list=[]
        x_list_error=[]
        y_list_error=[]
        fig, ax = plt.subplots()
        ax.fill_between(self.x, self.q*(1.-q_uncertainty), self.q*(1.+q_uncertainty), color='blue', alpha=.2)
        ax.axvline(x_stability_boundary_min,color='orange',alpha=0.6)
        ax.axvline(x_stability_boundary_max,color='orange',alpha=0.6)
        ax.plot(self.x,self.q,color='blue',label=r'Safety factor $q_0$')
        
        stable_counter=0
        unstable_counter=0
        for i in range(len(n_list)):
            n=n_list[i]

            x1=[]
            y1=[]
            x1_error=[]
            y1_error=[]

            qmin = np.min(self.q*(1.-q_uncertainty))
            qmax = np.max(self.q*(1.+q_uncertainty))
        
            m_min = math.ceil(qmin*n)
            m_max = math.floor(qmax*n)
            m_list=np.arange(m_min,m_max+1)

            for m in m_list:
                surface=float(m)/float(n)
                if np.min(q_range)*(1.-q_uncertainty)<surface and surface<np.max(q_range)*(1.+q_uncertainty):
                    x1.append(0.5*(x_stability_boundary_max+x_stability_boundary_min))
                    y1.append(surface)
                    x1_error.append(0.5*(x_stability_boundary_max-x_stability_boundary_min))
                    y1_error.append(0)

            if unstable_list[i]==1:#for unstable case
                if unstable_counter==0:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='red',ms=marker_size,label='Unstable')
                else:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='red',ms=marker_size)
                unstable_counter=unstable_counter+1
            elif unstable_list[i]==0:#for stable case
                if stable_counter==0:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='green',ms=marker_size,label='Stable')
                else:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='green',ms=marker_size)
                stable_counter=stable_counter+1
            x_list.append(x1)
            y_list.append(y1)
            x_list_error.append(x1_error)
            y_list_error.append(y1_error)

        ax.plot(self.x,self.q*q_scale+q_shift,color='orange',label=r'Modified $q=$'+str(q_scale)+r'$*q_0$')
        ax.set_xlim(self.x0_min,self.x0_max)
        ax.set_xlabel(r'$\rho_{tor}$')
        ax.set_ylabel('Safety factor')
        plt.legend(loc='upper left')
        plt.show()


    #this attribute only plot the nearest rational surface for the given toroidal mode number
    def Plot_ome_q_surface_demo(self,peak_percent,n_min,n_max,f_min,f_max,with_doppler=False):
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
            Unstabel_surface_counter=0
            try:
                x0,m=self.Rational_surface_peak_surface(n)
            except:
                continue

            if x_min_plot<x0 and x0<x_max_plot:
                if x_min<x0 and x0<x_max:
                    host.axvline(x0,color='orange',alpha=1)
                    if with_doppler:
                        host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                    else:
                        host.scatter([x0],[float(n)*(self.ome)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                    Unstabel_surface_counter=Unstabel_surface_counter+1
                else:
                    host.axvline(x0,color='orange',alpha=0.3)
                    if with_doppler:
                        host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                    else:
                        host.scatter([x0],[float(n)*(self.ome)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
            
                    
            if with_doppler:
                if Unstabel_surface_counter>0:
                    p1, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
                else:
                    p2, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "k-", label=r'Stable $\omega_{*e}$')
            else:
                if Unstabel_surface_counter>0:
                    p1, = host.plot(self.x,(self.ome)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
                else:
                    p2, = host.plot(self.x,(self.ome)*float(n), "k-", label=r'Stable $\omega_{*e}$')
        

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
        
        lines = [p1,p2]
        
        #host.legend(lines, ['Unstable area',r'Unstable $\omega_{*e}$',r'Stable $\omega_{*e}$'])
        host.legend(lines, [l.get_label() for l in lines])
        plt.show()


    def Plot_ome_q_surface_demo_no_box(self,n_min,n_max,n_unstable,with_doppler=False):
        n_list=np.array(np.arange(n_min,n_max+1),dtype=int)
        x_min=np.min(self.x)
        x_max=np.max(self.x)
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        for n in n_list:
            Unstabel_surface_counter=0
            try:
                x0,m=self.Rational_surface_peak_surface(n)
            except:
                continue

            
            if n in n_unstable:
                host.axvline(x0,color='orange',alpha=1)
                if with_doppler:
                    host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                else:
                    host.scatter([x0],[float(n)*(self.ome)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                Unstabel_surface_counter=Unstabel_surface_counter+1
            else:
                host.axvline(x0,color='orange',alpha=0.3)
                if with_doppler:
                    host.scatter([x0],[float(n)*(self.ome+self.Doppler)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                else:
                    host.scatter([x0],[float(n)*(self.ome)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
            
            if with_doppler:
                if n in n_unstable:
                    p1, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
                else:
                    p2, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "k-", label=r'Stable $\omega_{*e}$')
            else:
                if n in n_unstable:
                    p1, = host.plot(self.x,(self.ome)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
                else:
                    p2, = host.plot(self.x,(self.ome)*float(n), "k-", label=r'Stable $\omega_{*e}$')

        
        host.set_xlim(np.min(self.x),np.max(self.x))
        #host.set_ylim(0, np.max(f_max)*1.2)
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
        
        try:
            lines = [p1,p2]
        except:
            try:
                lines = [p2]
            except:
                lines = [p1]
        
        #host.legend(lines, ['Unstable area',r'Unstable $\omega_{*e}$',r'Stable $\omega_{*e}$'])
        host.legend(lines, [l.get_label() for l in lines])
        plt.show()


    def Plot_ome_q_surface_frequency_list(self,peak_percent,n_min,n_max,\
                Frequency_list,Frequency_error=0.2,save_imag=False,\
                image_name='./plot.jpg'):
        n_list=np.array(np.arange(n_min,n_max+1),dtype=int)
        x_peak,x_min,x_max=self.ome_peak_range(peak_percent)
        x_peak_plot,x_min_plot,x_max_plot=self.ome_peak_range(peak_percent*10.)

        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        #par1 = host.twinx()
        #p1, = par1.plot(self.x, self.q, "b-", label='Safety Factor')
        for f in Frequency_list:
            f_min=f*(1.-Frequency_error)
            f_max=f*(1.+Frequency_error)
            x_fill=[x_min,x_min,x_max,x_max]
            y_fill=[f_min,f_max,f_max,f_min]
            #matplotlib.patches.Rectangle((0,total_trans-trans_error),10,2.*trans_error,alpha=0.4)
            p1, = host.fill(x_fill,y_fill,color='blue',alpha=0.3,label='Unstable area')
        
        
        for n in n_list:
            Unstabel_surface_counter=0
            try:
                x0,m=self.Rational_surface_peak_surface(n)
            except:
                continue
            if x_min_plot<x0 and x0<x_max_plot:
                if x_min<x0 and x0<x_max:
                    host.axvline(x0,color='orange',alpha=1)
                    host.scatter([x0],[float(n)*(self.ome)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
                    Unstabel_surface_counter=Unstabel_surface_counter+1
                else:
                    host.axvline(x0,color='orange',alpha=0.3)
                    host.scatter([x0],[float(n)*(self.ome)[np.argmin(abs(self.x-x0))]],s=30,color='blue')
            
                    
            if Unstabel_surface_counter>0:
                p2, = host.plot(self.x,(self.ome)*float(n), "r-", label=r'Unstable $\omega_{*e}$')
            else:
                p2, = host.plot(self.x,(self.ome)*float(n), "k-", label=r'Stable $\omega_{*e}$')


        
        f_max=np.max(Frequency_list)
        host.set_xlim(np.min(self.x),np.max(self.x))
        host.set_ylim(0, np.max(f_max)*(1+Frequency_error)*1.2)
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
        
        lines = [p1,p2]
        
        #host.legend(lines, ['Unstable area',r'Unstable $\omega_{*e}$',r'Stable $\omega_{*e}$'])
        host.legend(lines, [l.get_label() for l in lines])
        if save_imag==True:
            plt.savefig(image_name)#save the 
        else:
            plt.show()

        plt.close(fig)


    def Plot_ome_q_stability_boundary(self,peak_percent):
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)

        x_peak,x_min,x_max=self.ome_peak_range(peak_percent)
        par1 = host.twinx()
        x=self.x
        y1=self.ome
        y3=self.ome+self.Doppler
        print(self.Doppler)
        y2=self.q
        p1, = host.plot(x, y1, "b-", label=r'$\omega_{*e}$')
        p2, = host.plot(x, y3, "g-", label=r'$\omega_{*e}$ in lab frame')
        p3, = par1.plot(x, y2, "r-", label=r'safety factor')
        p4, = par1.plot([x_min]*1000,np.arange(0,100,(100./1000.)),color='orange',alpha=0.6, label='Unstable range')
        par1.axvline(x_max,color='orange',alpha=0.6)
        
        host.set_xlim(self.x0_min, self.x0_max)
        host.set_ylim(0., np.max(self.ome+self.Doppler)*1.2)
        par1.set_ylim(np.min(self.q)*0.7, np.max(self.q)*1.2)
        
        host.set_xlabel(r"$\rho_{tor}$")
        host.set_ylabel(r"$\omega_{*e}$(kHz)")
        par1.set_ylabel("Safety factor")
        
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p3.get_color())
        
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
        #par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [p1, p2, p3, p4]
        
        host.legend(lines, [l.get_label() for l in lines], loc='upper left')
        plt.show()


    def Plot_highlight_top_percent_ome(self,peak_percent,n_min,n_max,f_min,f_max):
        n_list=np.array(np.arange(n_min,n_max+1),dtype=int)
        x_peak,x_min,x_max=self.ome_peak_range(peak_percent)
        plt.clf()
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        fig.set_size_inches(8, 6)
            
        x_min=x_min
        x_max=x_max
        y_min=0
        y_max=10000
        x_fill=[x_min,x_min,x_max,x_max]
        y_fill=[y_min,y_max,y_max,y_min]
        

        p1,=host.plot(self.x,(self.ome+self.Doppler)*float(1), "b-",alpha=0.5, label=r'$\omega_{*e}$')
        p2,=host.fill(x_fill,y_fill,color='purple',alpha=0.1,label=r'$\mu/x_{*} \leq 0.3$')
        
        host.set_xlim(np.min(self.x),np.max(self.x))
        host.set_ylim(0, np.max(self.ome+self.Doppler)*1.2)
        
        host.set_xlabel(r"$\rho_{tor}$")
        host.set_ylabel(r"$\omega_{*e}$(kHz)")
        
        
        host.yaxis.label.set_color('black')
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors='black', **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [ p1,p2]
        
        host.legend(lines, [l.get_label() for l in lines])
        plt.savefig('./define_mu.png')
        plt.show()

    def std_SLiM_calc(self,n_min=1,n_max=8,surface_num=2,Run_mode=6,peak_percent=0.2,manual_fit=False,NN_path='./SLiM_NN/Trained_model'):
        x_list=[]
        n_list=[]
        m_list=[]
        nu_list=[]
        zeff_list=[]
        eta_list=[]
        shat_list=[]
        beta_list=[]
        ky_list=[]
        ModIndex_list=[]
        mu_list=[]
        omega_e_plasma_list=[]
        omega_e_lab_list=[]
        omega_n_kHz_list=[]
        omega_n_cs_a_list=[]
        xstar_list=[]
        q_scale_list0=[]
        q_shift_list0=[]

        self.ome_peak_range(peak_percent)
        mean_rho,xstar=self.omega_gaussian_fit(manual=manual_fit)
        self.set_xstar(xstar)
        
        if Run_mode==1:#simple rational surface alignment
            ModIndex=-1
            filename='./rational_surface_alignment.csv'
        if Run_mode==2 or Run_mode==4 or Run_mode==5 or Run_mode==6:#global dispersion
            ModIndex=1
            filename='./global_dispersion.csv'
        elif Run_mode==3:#local dispersion
            ModIndex=0
            filename='./local_dispersion.csv'
        
        
        peak_index=np.argmin(abs(self.x-self.x_peak))
        omega_e_peak_kHz=self.ome[peak_index]
        
        cs_to_kHz=self.cs_to_kHz[peak_index]
        print('Finding the rational surfaces')
        n0_list=np.arange(n_min,n_max+1,1)

        for n in tqdm(n0_list):
            x_surface_near_peak_list, m_surface_near_peak_list=self.Rational_surface_top_surfaces(n,top=surface_num)
            print(x_surface_near_peak_list)
            print(m_surface_near_peak_list)
            for j in range(len(x_surface_near_peak_list)):
                x_surface_near_peak=x_surface_near_peak_list[j]
                m_surface_near_peak=m_surface_near_peak_list[j]
                if self.x_min<=x_surface_near_peak and x_surface_near_peak<=self.x_max:
                    nu,zeff,eta,shat,beta,ky,mu,xstar=\
                        self.parameter_for_dispersion(x_surface_near_peak,n)
                    factor=self.factor
                    index=np.argmin(abs(self.x-x_surface_near_peak))
                    omega_n_kHz=float(n)*self.omn[index]
                    omega_n_cs_a=float(n)*self.omn[index]/cs_to_kHz
                    omega_e_plasma_kHz=float(n)*self.ome[index]
                    omega_e_lab_kHz=float(n)*self.ome[index]+float(n)*self.Doppler[index]
                
                    n_list.append(n)
                    m_list.append(m_surface_near_peak)
                    x_list.append(x_surface_near_peak)
                    nu_list.append(nu)
                    zeff_list.append(zeff)
                    eta_list.append(eta)
                    shat_list.append(shat)
                    beta_list.append(beta)
                    ky_list.append(ky)
                    ModIndex_list.append(ModIndex)
                    mu_list.append(mu)
                    xstar_list.append(xstar)
                    omega_e_plasma_list.append(omega_e_plasma_kHz)
                    omega_e_lab_list.append(omega_e_lab_kHz)
                    omega_n_kHz_list.append(omega_n_kHz)
                    omega_n_cs_a_list.append(omega_n_cs_a)
                    q_scale_list0.append(self.q_scale)
                    q_shift_list0.append(self.q_shift)
        
        
        d = {'q_scale':q_scale_list0,'q_shift':q_shift_list0,\
            'n':n_list,'m':m_list,'rho_tor':x_list,\
            'omega_plasma_kHz':[0]*len(n_list),\
            'omega_lan_kHz':[0]*len(n_list),\
            'gamma_cs_a':[0]*len(n_list),\
            'omega_n_kHz':omega_n_kHz_list,\
            'omega_n_cs_a':omega_n_cs_a_list,\
            'omega_e_plasma_kHz':omega_e_plasma_list,\
            'omega_e_lab_kHz':omega_e_lab_list,\
            'peak_percentage':omega_e_plasma_list/\
                    (omega_e_peak_kHz*np.array(n_list,dtype=float)),\
            'nu':nu_list,'zeff':[zeff]*len(n_list),'eta':eta_list,\
            'shat':shat_list,'beta':beta_list,'ky':ky_list,\
            'ModIndex':ModIndex_list,'mu':mu_list,'xstar':xstar_list}
        df=pd.DataFrame(d)   #construct the panda dataframe
    
        if Run_mode==6:
            print('**************')
            print('**************')
            print(Run_mode)
            if hasattr(self, 'Dispersion_NN_obj'):
                pass
            else:
                self.Dispersion_NN_obj=Dispersion_NN(NN_path)

        
        if Run_mode==1:
            df_calc=0
            df_para=df
        else:    
            with open(filename, 'w', newline='') as csvfile:     #clear all and then write a row
                data = csv.writer(csvfile, delimiter=',')
                data.writerow(['q_scale','q_shift','n','m','rho_tor',\
                    'omega_plasma_kHz','omega_lab_kHz',\
                    'gamma_cs_a','omega_n_kHz',\
                    'omega_n_cs_a','omega_e_plasma_kHz',\
                    'omega_e_lab_kHz','peak_percentage',\
                    'nu','zeff','eta','shat','beta','ky',\
                    'ModIndex','mu','xstar'])
            csvfile.close()

            print('Calculate the dispersion relations')
            
            for i in tqdm(range(len(df['nu']))):
                if Run_mode==4:
                    w0=self.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                        df['shat'][i],df['beta'][i],df['ky'][i],\
                        df['ModIndex'][i],df['mu'][i],df['xstar'][i],manual=True)
                elif Run_mode==5:
                    w0=self.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                        df['shat'][i],df['beta'][i],df['ky'][i],\
                        df['ModIndex'][i],df['mu'][i],df['xstar'][i],manual=5)
                elif Run_mode==6:
                    w0=self.Dispersion_NN_obj.Dispersion_omega(df['nu'][i],df['zeff'][i],df['eta'][i],\
                            df['shat'][i],df['beta'][i],df['ky'][i],df['mu'][i],df['xstar'][i])
                else:
                    w0=self.Dispersion(df['nu'][i],df['zeff'][i],df['eta'][i],\
                        df['shat'][i],df['beta'][i],df['ky'][i],\
                        df['ModIndex'][i],df['mu'][i],df['xstar'][i])
                
                omega=np.real(w0)
                omega_kHz=omega*omega_n_kHz_list[i]
                gamma=np.imag(w0)
                gamma_cs_a=gamma*omega_n_cs_a_list[i]

                with open(filename, 'a+', newline='') as csvfile: #adding a row
                    data = csv.writer(csvfile, delimiter=',')
                    data.writerow([ q_scale_list0[i],\
                        q_shift_list0[i],\
                        df['n'][i],df['m'][i],df['rho_tor'][i],\
                        omega_kHz,\
                        omega_kHz+df['omega_e_lab_kHz'][i]-df['omega_e_plasma_kHz'][i],\
                        gamma_cs_a,\
                        df['omega_n_kHz'][i],df['omega_n_cs_a'][i],\
                        df['omega_e_plasma_kHz'][i],df['omega_e_lab_kHz'][i],
                        df['peak_percentage'][i],df['nu'][i],\
                        df['zeff'][i],df['eta'][i],\
                        df['shat'][i],df['beta'][i],df['ky'][i],\
                        df['ModIndex'][i],df['mu'][i],df['xstar'][i] ])
                csvfile.close()
            df_calc=pd.read_csv(filename)
            df_para=df
        return df_calc,df_para

