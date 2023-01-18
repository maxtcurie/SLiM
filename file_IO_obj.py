import matplotlib.pyplot as plt
import sys
import os
from glob import glob
import numpy as np
sys.path.insert(1, './Tools')

from read_profiles import read_profile_file
from read_profiles import read_geom_file
from read_EFIT_file import get_geom_pars
from read_pfile import read_pfile_to_dict
from read_pfile import psi_rhot
from write_iterdb import output_iterdb
from write_pfile import write_pfile
from write_gfile import mod_q_gfile
from interp import interp


def dir_scaner(base_dir,level=1):   
    base_dir=base_dir+'/'
    for i in range(level):
        base_dir=base_dir+"*/"
    directory=glob(base_dir, recursive = True)
    
    dir_list=directory
    return dir_list


def get_g_and_p_file(dir_name):
    file_list=os.listdir(dir_name)
    gfile=''
    pfile=''
    for file in file_list:
        if file[:2]=='g1':
            gfile=dir_name+file
        elif file[:2]=='p1':
            pfile=dir_name+file

    return gfile,pfile

class file_IO_obj:
    profile_type=('pfile','ITERDB')
    geomfile_type=('gfile','GENE_tracor')
    def __init__(self,profile_type,profile_name,\
                    geomfile_type,geomfile_name,\
                    suffix='.dat'):
        self.profile_name=profile_name
        self.geomfile_name=geomfile_name
        self.suffix=suffix

        if profile_type not in file_IO_obj.profile_type:
            raise ValueError(f'{profile_type} is not a valid profile type, need to be pfile or ITERDB')
        else: 
            self.profile_type  = profile_type

        if geomfile_type not in file_IO_obj.geomfile_type:
            raise ValueError(f'{geomfile_type} is not a valid geomfile_type, need to be gfile or GENE_tracor')
        else: 
            self.geomfile_type  = geomfile_type

        #*************Loading the data******************************************
        if self.profile_type=="ITERDB":
            rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 \
                = read_profile_file(self.profile_type,\
                                    self.profile_name,\
                                    self.geomfile_name,\
                                    self.suffix)
        elif self.profile_type=="pfile":
            rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 \
                = read_profile_file(self.profile_type,\
                                    self.profile_name,\
                                    self.geomfile_name,\
                                    self.suffix)

        if self.geomfile_type=="gfile": 
            xgrid, q, R_ref= read_geom_file(self.geomfile_type,\
                                            self.geomfile_name,\
                                            self.suffix)
        elif self.geomfile_type=="GENE_tracor":
            xgrid, q, Lref, R_ref, Bref, x0_from_para\
                 = read_geom_file(self.geomfile_type,\
                                    self.geomfile_name,\
                                    self.suffix)


        uni_rhot = np.linspace(min(rhot0),max(rhot0),2000)

        self.x    = uni_rhot
        self.rhot = uni_rhot
        self.rhop = np.interp(uni_rhot,rhot0,rhop0)
        self.  te = np.interp(uni_rhot,rhot0,te0)
        self.  ti = np.interp(uni_rhot,rhot0,ti0)
        self.  ne = np.interp(uni_rhot,rhot0,ne0)
        self.  ni = np.interp(uni_rhot,rhot0,ni0)
        self.  nz = np.interp(uni_rhot,rhot0,nz0)
        self.vrot = np.interp(uni_rhot,rhot0,vrot0)
        

        self.xgrid=uni_rhot
        self.q=    np.interp(uni_rhot,xgrid,q)
        self.R_ref=R_ref


    def output_profile(self,profile_type,profile_name,shot_num=999999,time_str='1000'):
        if profile_type not in file_IO_obj.profile_type:
            raise ValueError(f'{profile_type} is not a valid profile type, need to be pfile or ITERDB')
        else: 
            pass
        if profile_type=="ITERDB":
            self.output_ITERDB(profile_name,shot_num,time_str)
        elif profile_type=="pfile":
            self.output_pfile(profile_name)

    def output_geofile(self,geomfile_type,geomfile_name):
        if geomfile_type not in file_IO_obj.geomfile_type:
            raise ValueError(f'{geomfile_type} is not a valid profile type, need to be pfile or ITERDB')
        else: 
            pass
        if geomfile_type=="gfile":
            self.output_gfile(geomfile_name)
        elif geomfile_type=="GENE_tracor":
            print('Stay tunned for the GENE_tracor output capability. ')


    def output_pfile(self,output_profile_name):
        pfile_dic=read_pfile_to_dict(self.profile_name)
        
        psi,rhot=psi_rhot(self.geomfile_name)

        x = self.x
        ne  =self.ne*1.E-20
        te  =self.te*1.E-3
        vrot=self.vrot

        needed_quant={'ne':ne,
                    'te':te,
                    'vtor':vrot}
        needed_quant_key=list(needed_quant.keys())

        #obj_* is the profile generated inside of the obj, i.e. output profile
        #dic_* is the profile in the dictionary readed from p file, i.e. nominal profile

        keys_name_in_dic={}
        for i in range(len(needed_quant)):
            for key in pfile_dic.keys():
                if needed_quant_key[i] in key:
                    f_tmp=needed_quant[needed_quant_key[i]]
                    df_tmp=np.gradient(f_tmp,self.x)
                    keys_name_in_dic[needed_quant_key[i]]={'name':key,\
                                                        'dic_psi':pfile_dic[key]['psi'],\
                                                        'obj_x':self.x,\
                                                        'dic_f':pfile_dic[key]['f'],\
                                                        'obj_f':f_tmp,\
                                                        'dic_df':pfile_dic[key]['df'],\
                                                        'obj_df':df_tmp
                                                        }
        for key in needed_quant_key:
            p_x_psi=keys_name_in_dic[key]['dic_psi']
            output_x_rho= keys_name_in_dic[key]['obj_x']
            output_f_rho= keys_name_in_dic[key]['obj_f']
            output_df_rho=keys_name_in_dic[key]['obj_df']
            
            p_x_rhot=np.interp(p_x_psi,psi,rhot)

            new_dic_f =np.interp(p_x_rhot,output_x_rho,output_f_rho)
            new_dic_df =np.interp(p_x_rhot,output_x_rho,output_df_rho)

            dic_key=keys_name_in_dic[key]['name']
            print(key)
            print(dic_key)
            
            pfile_dic[dic_key]['f']=new_dic_f
            pfile_dic[dic_key]['df']=new_dic_df

        write_pfile(output_profile_name,pfile_dic)
        
    
    def output_gfile(self,output_geofile_name):
        mod_q_gfile(self.geomfile_name,output_geofile_name,self.x,self.q)
        
    

    def output_ITERDB(self,out_iterdb_name,shot_num=999999,time_str='1000'):
        output_iterdb(self.rhot,self.rhop,self.ne*1.E-19,\
                        self.te*1.E-3,self.ni*1.E-19,\
                        self.ti*1.E-3,out_iterdb_name,\
                        shot_num,time_str,\
                        vrot=self.vrot,nimp=self.nz*1.E-19)


    def find_pedestal(self,x,p):

        if np.min(x)<0.65:
            num=int(len(x)*0.7)
            x=(x[num:]).copy()
            p=(p[num:]).copy()
        dp=np.gradient(p,x)  #Second order of pressure
        ddp   = np.gradient(dp,x)  #Second order of pressure
        midped = x[np.argmin(dp)]
        topped = x[np.argmin(ddp)]
        return topped,midped


    def modify_profile(self,\
                        q_scale=1.,q_shift=0.,\
                        shat_scale=1.,\
                        ne_scale=1.,te_scale=1.,\
                        ne_shift=0.,te_shift=0.,\
                        Doppler_scale=1.,\
                        show_plot=False):
        #************start of modification***********
        #********************************************

        #**********************************
        #*************modify ne************
        try:
            self.ne_weight
        except:
            ne_top_ped,ne_mid_ped=self.find_pedestal(self.x,self.ne)
            self.ne_mid_ped=ne_mid_ped
            ne_width=ne_mid_ped-ne_top_ped
            self.ne_weight = 0.5+0.5*np.tanh((self.x-ne_top_ped)/ne_width)
        

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
            self.te_weight = 0.5+0.5*np.tanh((self.x-te_top_ped)/te_width)

        index=np.argmin(abs(self.x-self.te_mid_ped))
        te_mod=self.te*(1.+(te_scale-1.)*self.te_weight)+(te_shift)*self.te[index]
        #*************modify Te************
        #**********************************

        mid_ped,top_ped=self.find_pedestal(self.x,self.te)


        width=(mid_ped-top_ped)*10.
        try:
            self.shat_weight
        except:
            self.shat_weight = -1.*np.tanh((self.x-mid_ped)/width)/2
        
        q_mod=self.q*q_scale*(1.+(shat_scale-1.)*self.shat_weight)+q_shift

        vrot_mod=self.vrot*Doppler_scale

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
            ax[3].plot(self.x,self.vrot)
            ax[3].plot(self.x,vrot_mod)
            ax[3].set_ylabel(r'$vrot$',fontsize=10)
            ax[3].set_xlabel(r'$\rho_{tor}$',fontsize=10)
            plt.tight_layout()
            plt.show()

        #*************end of modification************
        #********************************************

        self.x =self.x 
        self.ne=ne_mod      # in 10^19 /m^3
        self.te=te_mod     # in keV 
        self.q=q_mod
        self.vrot=vrot_mod
