import csv
import numpy as np
import math
import matplotlib.pyplot as plt

from max_pedestal_finder import find_pedestal
from read_iterdb_file import read_iterdb_file
from read_pfile import p_to_iterdb_format
from read_EFIT import read_EFIT
from read_write_geometry import read_geometry_global
from parIOWrapper import init_read_parameters_file
#from max_profile_reader import profile_e_info
#from max_profile_reader import profile_i_info
from interp import interp
#Created by Max T. Curie  11/02/2020
#Last edited by Max Curie 11/02/2020
#Supported by scripts in IFS

def input():
    temp=input("The profile file type: 1. ITERDB  2. pfile")
    suffix="1"
    if temp==1:
        profile_type="ITERDB"    # "ITERDB" "pfile"  "GENE"
    elif temp==2:
        profile_type="pfile"    # "ITERDB" "pfile"  "GENE"
    else:
        print("Please type 1 or 2")

    profile_name = input("The profile file name: ")


    temp=input("The geometry file type: 1. gfile  2. GENE_tracor")
    if temp==1:
        geomfile_type="gfile"    
    elif temp==2:
        geomfile_type="GENE_tracor"  
        suffix=input("Suffix (0001, 1, dat): ")
    else:
        print("Please type 1 or 2")

    geomfile_name = input("The geometry file name: ")   

    return profile_type, geomfile_type, profile_name, geomfile_name, suffix
    

def read_profile_file(profile_type,profile_name,geomfile_name,suffix='.dat'):
    if profile_type=="ITERDB":
        rhot0, te0, ti0, ne0, ni0, nz0, vrot0 = read_iterdb_file(profile_name)
        psi0 = np.linspace(0.,1.,len(rhot0))
        rhop0 = np.sqrt(np.array(psi0))
        return rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0

    elif profile_type=="pfile":
        rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 = p_to_iterdb_format(profile_name,geomfile_name)
        return rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0

    elif profile_type=="profile_e":
        if suffix=="dat":
            suffix=".dat"
        else:
            suffix="_"+suffix
        rhot0, te0, ti0, ne0, ni0, nz0, vrot0 = read_iterdb_file(profile_name)
        psi0 = np.linspace(0.,1.,len(rhot0))
        rhop0 = np.sqrt(np.array(psi0))

        x_a,x_rho_ref,T,n0,omt,omn = profile_e_info(suffix)
        if 1==0:
            plt.clf()
            plt.plot(x_a,n0*1.0e19,label='profile')
            plt.plot(rhot0,ne0,label='ITERDB')
            #plt.plot(x_rho_ref,T,label='x_rho_ref')
            plt.legend()
            plt.show()

            plt.clf()
            plt.plot(x_a,T*1000.,label='profile')
            plt.plot(rhot0,te0,label='ITERDB')
            #plt.plot(x_rho_ref,T,label='x_rho_ref')
            plt.legend()
            plt.show()
        #x_a,x_rho_ref,T,n0,omt,omn = profile_i_info(suffix)
        
        rhot0_range_min=np.argmin(abs(rhot0-x_a[0]))
        rhot0_range_max=np.argmin(abs(rhot0-x_a[-1]))
        rhot=rhot0[rhot0_range_min:rhot0_range_max]
        rhop=rhop0[rhot0_range_min:rhot0_range_max]
        ti=ti0[rhot0_range_min:rhot0_range_max]
        ni=ni0[rhot0_range_min:rhot0_range_max]
        vrot=vrot0[rhot0_range_min:rhot0_range_max]

        rhot0 = np.linspace(min(rhot),max(rhot),len(rhot)*3.)
        
        rhop0=interp(rhot,rhop,rhot0)
        te0 = interp(x_a,T*1000.,rhot0)
        ti0=interp(rhot,ti,rhot0)
        ni0=interp(rhot,ni,rhot0)
        nz0 = interp(rhot,nz,rhot0)
        ne0 = interp(x_a,n0*1.0e19,rhot0)

        vrot0=interp(rhot,vrot,rhot0)
        
        print("**************************")
        print("**************************")
        print("x_a min max: "+str(np.min(x_a))+", "+str(np.max(x_a)))
        print("**************************")
        print("**************************")
        print("**************************")

        return rhot0, rhop0, te0, ti0, ne0, ni0, vrot0

def read_geom_file(file_type,file_name,suffix="dat"):
    if file_type=="gfile":
        EFITdict = read_EFIT(file_name)
        
        xgrid = EFITdict['rhotn'] #rho_tor

        q = EFITdict['qpsi']
        R = EFITdict['R']
        print("**************************")
        print("**************************")
        print(str(R))
        print(np.shape(R))
        R_ref=R[int(len(R)/2)]
        print("**************************")
        print("**************************")
        return xgrid, q, R_ref
        
    elif file_type=="GENE_tracor":
        gpars,geometry = read_geometry_global(file_name)
        q=geometry['q']
        R=geometry['geo_R']
        print("**************************")
        print("**************************")
        print(str(R))
        print(np.shape(R))
        (nz0,nx0)=np.shape(R)
        R_ref=R[int(nz0/2),int(nx0/2)]
        print("**************************")
        print("**************************")
        
        if suffix=="dat":
            suffix=".dat"
        else:
            suffix="_"+suffix

        pars = init_read_parameters_file(suffix)
        Lref=pars['Lref']

        Bref=pars['Bref']
        x0_from_para=pars['x0']
        #xgrid=rho_tor
        if 'lx_a' in pars:
            xgrid = np.arange(pars['nx0'])/float(pars['nx0']-1)*pars['lx_a']+pars['x0']-pars['lx_a']/2.0
        else:
            xgrid = np.arange(pars['nx0'])/float(pars['nx0']-1)*pars['lx'] - pars['lx']/2.0
        print("**************************")
        print("**************************")
        print("xgrid min max: "+str(np.min(xgrid))+", "+str(np.max(xgrid)))
        print("**************************")
        print("**************************")
        print("**************************")
        return xgrid, q, Lref, R_ref, Bref, x0_from_para
    
def profile_e_info(suffix):
    gene_e = 'profiles_e'+suffix
    gene_i = 'profiles_i'+suffix
    gene_z = 'profiles_z'+suffix
    suffix=gene_e
    f = open(suffix, 'r')
    #prof=f.read()#the read from the profile
    prof = np.genfromtxt(suffix, dtype=float, skip_header=2)
    x_a=prof[:,0]
    x_rho_ref=prof[:,1]
    T=prof[:,2]
    n0=prof[:,3]
    omt=prof[:,4]
    omn=prof[:,5]

    return x_a,x_rho_ref,T,n0,omt,omn 

def profile_i_info(suffix):
    gene_e = 'profiles_e'+suffix
    gene_i = 'profiles_i'+suffix
    gene_z = 'profiles_z'+suffix
    suffix=gene_i
    f = open(suffix, 'r')
    #prof=f.read()#the read from the profile
    prof = np.genfromtxt(suffix, dtype=float, skip_header=2)
    x_a=prof[:,0]
    x_rho_ref=prof[:,1]
    T=prof[:,2]
    n0=prof[:,3]
    omt=prof[:,4]
    omn=prof[:,5]

    return x_a,x_rho_ref,T,n0,omt,omn 

def profile_z_info(suffix):
    gene_e = 'profiles_e'+suffix
    gene_i = 'profiles_i'+suffix
    gene_z = 'profiles_z'+suffix
    suffix=gene_z
    f = open(suffix, 'r')
    #prof=f.read()#the read from the profile
    prof = np.genfromtxt(suffix, dtype=float, skip_header=2)
    x_a=prof[:,0]
    x_rho_ref=prof[:,1]
    T=prof[:,2]
    n0=prof[:,3]
    omt=prof[:,4]
    omn=prof[:,5]

    return x_a,x_rho_ref,T,n0,omt,omn 