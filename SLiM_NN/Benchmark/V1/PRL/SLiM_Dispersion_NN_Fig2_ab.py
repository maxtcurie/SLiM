import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys as sys
import pandas as pd
sys.path.insert(1, './../..')

from Dispersion_NN import Dispersion_NN

#************Start of of user block******************
output_csv_file='./Fig2_ab_NN.csv'

read_csv=True	#change to True if one want to read the data from csv file,\
				# instead of calculating from SLiM_NN
read_output_csv_file='./Fig2_ab_NN.csv'

fontsize=12

nu=1.
zeff=1.
eta=2.
shat=0.006
beta=0.002
ky=0.0
mu=0.
xstar=10

mu_list=np.arange(0,6,0.01)

path='./../../Trained_model/'
NN_omega_file	   =path+'SLiM_NN_omega.h5'
NN_gamma_file	   =path+'SLiM_NN_stabel_unstable.h5'
norm_omega_csv_file=path+'NN_omega_norm_factor.csv'
norm_gamma_csv_file=path+'NN_stabel_unstable_norm_factor.csv'

#************End of user block******************

para_list=[[nu, zeff, eta, shat,  beta,  ky,   mu, xstar]\
			for mu in mu_list]

Dispersion_NN_obj=Dispersion_NN(NN_omega_file,NN_gamma_file,norm_omega_csv_file,norm_gamma_csv_file)

f_list=[]
gamma_list=[]
gamma_10_list=[]

if read_csv==True:
	data=pd.read_csv(read_output_csv_file)  	#data is a dataframe. 
	
	mu_list=data['mu']
	f_list=data['f']
	gamma_list=data['gamma']
	gamma_10_list=data['gamma_10']

else:
	nu_output_list=[]
	zeff_output_list=[]
	eta_output_list=[]
	shat_output_list=[]
	beta_output_list=[]
	ky_output_list=[]
	mu_output_list=[]
	xstar_output_list=[]
	
	for para in tqdm(para_list):
		[nu, zeff, eta, shat,  beta,  ky,   mu, xstar]=para
		nu_output_list.append(nu)
		zeff_output_list.append(zeff)
		eta_output_list.append(eta)
		shat_output_list.append(shat)
		beta_output_list.append(beta)
		ky_output_list.append(ky)
		mu_output_list.append(mu)
		xstar_output_list.append(xstar)
	
		w=Dispersion_NN_obj.Dispersion_omega(nu,zeff,eta,shat,beta,ky,mu,xstar)
		f_list.append(w.real)
		gamma_list.append(w.imag)
		if w.imag>0.00001:
			gamma_10_list.append(1)
		else:
			gamma_10_list.append(0)
	
	
	d = {'nu':nu_output_list, \
		'zeff':zeff_output_list,\
		'eta':eta_output_list,\
		'shat':shat_output_list,\
		'beta':beta_output_list, \
		'ky':ky_output_list,  \
		'mu':mu_output_list, \
		'xstar':xstar_output_list,\
		'f':f_list,'gamma':gamma_list,'gamma_10':gamma_10_list}
	df=pd.DataFrame(d)	#construct the panda dataframe
	df.to_csv(output_csv_file,index=False)

f_list=np.array(f_list)
gamma_list=np.array(gamma_list)
gamma_10_list=np.array(gamma_10_list)

if 1==1:
	mu = [0., 0.2, 0.4000000000000001, 0.6000000000000001, 
	  0.8, 1.0000000000000002, 1.2000000000000002, 1.4000000000000001,
	   1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 
	  3.8, 4., 4.2, 4.4, 4.6000000000000005, 4.8, 5., 5.2, 5.4, 
	  5.6000000000000005, 5.8, 6.]
	
	omega_GL=[3.3402221902282574 + 0.30351478126067694j, 3.339819027806459 + \
	        0.30251626161254197j, 3.3386113775562114 + \
	        0.299519150236142j, 3.3366048840481004 + \
	        0.29451883389021105j, 3.3338094050268188 + \
	        0.2875077845250888j, 3.330239702705062 + \
	        0.2784758510616202j, 3.325916437301237 + \
	        0.26741077771530514j, 3.320867470102928 + \
	        0.25429907786499767j, 3.315129449819169 + \
	        0.23912745457095902j, 3.3087495783082415 + \
	        0.22188503150436079j, 3.301787300682352 + \
	        0.20256672272320062j, 3.2943154078422054 + \
	        0.18117807576528636j, 3.2864196678833975 + \
	        0.15774176499488754j, 3.2781956988142795 + \
	        0.13230542610799947j, 3.269741640701579 + \
	        0.10494955633687897j, 3.2611458170489374 + \
	        0.07579286342392487j, 3.252470513418996 + \
	        0.04499146859743953j, 3.243735928307575 + \
	        0.012729144538328272j, 3.2349103531426673 - \
	        0.020800806238435793j, 3.225911217288539 - \
	        0.05541682355887535j, 3.21661670650449 - \
	        0.0909661497698298j, 3.2068827368269086 - \
	        0.12733274854397833j, 3.196558763969639 - \
	        0.16443610839982775j, 3.1854981704464587 - \
	        0.2022245827979803j, 3.1735619728046576 - \
	        0.24066752171038133j, 3.1606161239141173 - \
	        0.27974887836744705j, 3.146522634253652 - \
	        0.31946255660714284j, 3.131124206447458 - \
	        0.3598072525142266j, 3.114221794189359 - \
	        0.40077600669246904j, 3.095545358873761 - \
	        0.4423313336057385j, 3.0747267905520537 - 0.4843493276294521j]
	
	omega_LL= [3.4643694769568953 + \
	    0.24820717060813202j, \
	   3.462431572293245 + 0.2481162956019751j, \
	   3.4566245233354964 + 0.2478434742100395j, \
	   3.4469682841756475 + 0.24738812223105056j, \
	   3.4334959753201635 + 0.24674928233390014j, \
	   3.416253680544328 + 0.24592564835601405j, \
	   3.3953001651182757 + 0.24491559890626272j, \
	   3.370706517997846 + 0.2437172398616854j, \
	   3.3425557212585364 + 0.24232845523905625j, \
	   3.310942150696115 + 0.24074696581570124j, \
	   3.275971012117391 + 0.23897039477492243j, \
	   3.2377577183894526 + 0.23699633956073796j, \
	   3.1964272128000877 + 0.23482244904395702j, \
	   3.1521132446996694 + 0.23244650502292163j, \
	   3.1049576037450497 + 0.2298665070214878j, \
	   3.0551093193441408 + 0.22708075928937194j, \
	   3.0027238320936123 + 0.22408795886455454j, \
	   2.9479621441335317 + 0.22088728352744677j, \
	   2.8909899553805363 + 0.21747847845102936j, \
	   2.8319767925726045 + 0.21386194034440548j, \
	   2.771095137946891 + 0.2100387978899008j, \
	   2.7085195641916964 + 0.20601098729123363j, \
	   2.644425882054691 + 0.20178132178303107j, \
	   2.5789903066735507 + 0.1973535540009754j, \
	   2.5123886483092632 + 0.19273243017809993j, \
	   2.4447955327301463 + 0.18792373522001976j, \
	   2.3763836560105682 + 0.18293432781765745j, \
	   2.307323077996273 + 0.1777721648851293j, \
	   2.2377805581448893 + 0.17244631476023292j, \
	   2.167918936899749 + 0.16696695877148043j, \
	   2.0978965652056694 + 0.1613453809636113j]
	
	
	f_GL    =np.real(omega_GL)
	gamma_GL=np.imag(omega_GL)
	
	f_LL    =np.real(omega_LL)
	gamma_LL=np.imag(omega_LL)	



plt.clf()
plt.plot(mu,gamma_GL/(1.+eta),label='SLiM')
plt.plot(mu_list,gamma_10_list*np.max(gamma_GL/(1.+eta)),label='SLiM_NN')
plt.xlabel(r'$\mu$',fontsize=fontsize)
plt.ylabel(r'$\gamma/\omega_{*peak}$',fontsize=fontsize)
plt.axhline(0,color='black',alpha=0.5)
plt.grid(alpha=0.2)
plt.legend()
plt.show()

plt.clf()
plt.plot(mu,f_GL/(1.+eta),label='SLiM')
plt.plot(mu_list,f_list/(1.+eta),label='SLiM_NN')
plt.xlabel(r'$\mu$',fontsize=fontsize)
plt.ylabel(r'$\omega/\omega_{*peak}$',fontsize=fontsize)
plt.ylim(0,1.5)
plt.grid(alpha=0.2)
plt.legend()
plt.show()

