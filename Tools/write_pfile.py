#Created by Max T. Curie  03/29/2022
#Last edited by Max Curie 03/29/2022

#the input will be dictionary
def write_pfile(filename,p_quant_dict):
    file=open(filename,"w")
    
    for key in p_quant_dict.keys():
        header=p_quant_dict[key]['header']
        psi=p_quant_dict[key]['psi']
        f=p_quant_dict[key]['f']
        df=p_quant_dict[key]['df']

        write_quant(file,header,psi,f,df)
    
def write_quant(file,header,psi,f,df):
    file.write(header+'\n')
    for psi_i,f_i,df_i in zip(psi,f,df):
        file.write(' %.6f   %.6f   %.6f\n'%(psi_i,f_i,df_i))
