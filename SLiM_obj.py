import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import optimize
import math
import csv
import imageio
import sys as sys
sys.path.insert(1, './Tools')

from interp import interp
from finite_differences import fd_d1_o4
from Gaussian_fit import gaussian_fit
from Gaussian_fit import gaussian_fit_GUI
from max_pedestal_finder import find_pedestal_from_data
from max_pedestal_finder import find_pedestal
from read_profiles import read_profile_file
from read_profiles import read_geom_file
from read_EFIT_file import get_geom_pars
from max_stat_tool import Poly_fit
from max_stat_tool import coeff_to_Poly
from DispersionRelationDeterminantFullConductivityZeff import VectorFinder_auto
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
        print("suffix"+str(suffix))
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
        

        if zeff_manual==-1:
            zeff = (  ( ni+nz*(float(Z)**2.) )/ne  )[center_index]
        else:
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

        self.r_sigma=(1./shat)*( (me_SI/m_SI)**0.5 )
        self.R_ref=R_ref
        self.cs_to_kHz=gyroFreq/(2.*np.pi*1000.)
        self.omn=omega_n    #omega_n in kHz
        self.omn_nominal=omega_n
        self.cs=cref
        self.rho_s=rhoref*np.sqrt(2.)
        self.Lref=Lref
        self.x=uni_rhot
        self.shat=shat
        self.shat_nominal=shat
        self.eta=eta
        self.ky=ky
        self.ky_nominal=ky
        self.nu=nu
        self.zeff=zeff
        self.beta=beta
        self.q=q
        self.q_nominal=q
        self.ome=mtmFreq
        self.ome_nominal=mtmFreq
        self.Doppler=omegaDoppler

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
        

    def q_fit(self,order=5,show=False):
        dq= np.gradient(self.q,self.x)
        x_dq_min=self.x[np.argmin(abs(dq))]

        coeff=Poly_fit(self.x-x_dq_min, self.q, order, show)
        q_fit_coeff=coeff_to_Poly(self.x, x_dq_min, coeff, show)

        return coeff, q_fit_coeff

    def q_modify(self,q_scale,q_shift):
        q_mod=self.q*q_scale+q_shift

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
        return self.q

    def omega_gaussian_fit_GUI(self,root,x,data,rhoref,Lref):
        amplitude,mean,stddev=gaussian_fit_GUI(root,x,data)
        print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
        mean_rho=mean*Lref/rhoref         #normalized to rhoi
        xstar=abs(stddev*Lref/rhoref)
        
        popt=[0]*3
        popt[0] = amplitude
        popt[1] = mean     
        popt[2] = stddev   
    
        print(popt)
        print(mean_rho,xstar)
    
        return mean_rho,xstar
    
    def omega_gaussian_fit(self,manual=False):
        x=self.x 
        data=self.ome 
        rhoref=self.rho_s 
        Lref=self.Lref 

        amplitude,mean,stddev=gaussian_fit(x,data,manual)
        print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
        mean_rho=mean*Lref/rhoref         #normalized to rhoi
        xstar=abs(stddev*Lref/rhoref)
        
        popt=[0]*3
        popt[0] = amplitude
        popt[1] = mean     
        popt[2] = stddev   
    
        print(popt)
        print(mean_rho,xstar)
    
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
        print(surface_df)
        surface_df_sort=surface_df.sort_values(by='peak_distance')
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

    def Dispersion(self,nu,zeff,eta,shat,beta,ky,ModIndex,mu,xstar):
        w0=VectorFinder_auto(nu,zeff,eta,shat,beta,ky,ModIndex,abs(mu),xstar)
        return w0
    

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
    def Plot_ome_q_surface_demo(self,peak_percent,n_min,n_max,f_min,f_max):
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
            x0,m=self.Rational_surface_peak_surface(n)

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
                p2, = host.plot(self.x,(self.ome+self.Doppler)*float(n), "k-", label=r'Stable $\omega_{*e}$')

        

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

