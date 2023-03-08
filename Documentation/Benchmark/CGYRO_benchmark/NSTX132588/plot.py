import pandas as pd 
import matplotlib.pyplot as plt

SLiM_df=pd.read_csv('global_dispersion_3.csv')
CGYRO_df=pd.read_csv('output.csv')

print(CGYRO_df)

CGYRO_df=CGYRO_df[	CGYRO_df.modetype=='mtm' ]

	


ome=22.33400034/0.080044977
doppler=(12.25062738-17.52524309)/0.080044977
fig, ax=plt.subplots(nrows=2,ncols=1,sharex=True) 
ax[0].scatter(SLiM_df['ky'],SLiM_df['omega_e_lab_kHz'],label='SLiM')
ax[0].scatter(CGYRO_df['ky']/0.071856*0.08,\
			CGYRO_df['realfreq_over_omegastar']*(ome+doppler)*CGYRO_df['ky'],label='CGYRO')
ax[0].set_ylabel(r'$frequency (kHz)$',fontsize=15)
ax[0].legend()
ax[0].grid(alpha=0.2)

ax[1].scatter(SLiM_df['ky'],SLiM_df['gamma_cs_a'],label='SLiM')
ax[1].scatter(CGYRO_df['ky']/0.071856*0.08,CGYRO_df['growth_rate'],label='CGYRO')
ax[1].set_xlabel(r'$k_y\rho_s$',fontsize=15)
ax[1].set_ylabel(r'$\gamma (c_s/a)$',fontsize=15)
ax[1].set_xlim(0.05,0.35)
ax[1].set_ylim(0,0.1)
ax[1].grid(alpha=0.2)
plt.show()
