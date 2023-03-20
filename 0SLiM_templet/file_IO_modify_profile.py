import matplotlib.pyplot as plt

from SLiM_phys.file_IO_obj import file_IO_obj

#**********************************************
#**********Start of User block*****************
profile_type= "pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"
outputpath='./../Test_files/'
path='./../Test_files/'
profile_name=path+'p174819.03560'
geomfile_name=path+'g174819.03560'
suffix='_mod2'
plot_profile=True
plot_geofile=False
#************End of User Block*****************
#**********************************************

a=file_IO_obj(profile_type,profile_name,\
                geomfile_type,geomfile_name)

ne0=a.ne.copy()
te0=a.te.copy()
q0=a.q.copy()
x0=a.x.copy()


a.modify_profile(q_scale=0.95,q_shift=0.,\
                shat_scale=1.0,\
                ne_scale=0.8,te_scale=1.15,\
                ne_shift=0.,te_shift=0.,\
                Doppler_scale=1.,\
                show_plot=True)

a.output_profile("pfile",profile_name+suffix,shot_num=174819,time_str='03560')

#a.output_geofile("gfile",geomfile_name+suffix)



if plot_profile==True:
    plt.clf()
    plt.plot(x0,ne0,alpha=0.7,label='original')
    plt.plot(a.rhot,a.ne,alpha=0.7,label='modfied')
    plt.xlabel('rhot')
    plt.ylabel('ne(/m^3)')
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(x0,te0,alpha=0.7,label='original')
    plt.plot(a.rhot,a.te,alpha=0.7,label='modfied')
    plt.xlabel('rhot')
    plt.ylabel('Te(eV)')
    plt.legend()
    plt.show()

if plot_geofile==True:
    plt.clf()
    plt.plot(x0,q0,alpha=0.7,label='original')
    plt.plot(a.rhot,a.q,alpha=0.7,label='modfied')
    plt.legend()
    plt.show()

