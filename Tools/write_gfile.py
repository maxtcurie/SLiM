from read_profiles import read_geom_file
import numpy as np 


def mod_q_gfile(original_geofile_name,output_geofile_name,x_output,q_output,show_plot=False):
    xgrid, q_original, R_ref= read_geom_file('gfile',original_geofile_name,'.dat')

    f=open(original_geofile_name,'r')
    data = f.read()
    f.close()

    sdata = data.split('\n')

    #check the line # for different quantity 
    quant_dic={}
    for line,i in zip(sdata,range(len(sdata))):
        line=line.split()
        try:
            if '=' in line[1]:
                quant_dic[line[0]]=i
        except:
            pass

    keys=list(quant_dic.keys())
    
    print(keys)

    start_index=0
    end_index=0
    for key,i in zip(keys,range(len(keys))):
        if 'QPSI' == key:
            print(key)
            start_index=quant_dic[keys[i]]
            end_index=quant_dic[keys[i+1]]

    print('start_index='+str(start_index))
    print('end_index='+str(end_index))


    #read q
    if 1==0:
        q_data=[]
        for i in sdata[start_index].split()[2:]:
            q_data.append(float(i))
        for i in range(start_index+1,end_index):
            for j in sdata[i].split():
                q_data.append(float(j))



    #start of write the q
    file=open(output_geofile_name,"w")


    #before the q 
    for i in range(start_index):
        file.write(sdata[i]+'\n')

    q_output_tmp = np.interp(xgrid,x_output,q_output)

    #output q
    #first line
    file.write(' QPSI =    %.15f         %.15f      \n'%(q_output_tmp[0],q_output_tmp[1]))
    
    #the rest of the lines

    q_output_rest=q_output_tmp[2:]
    print('len(q_output_rest)')
    print(len(q_output_rest))

    if show_plot==True:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(xgrid,q_output_tmp,label='output')
        plt.plot(xgrid,q_original,label='from_gfile')
        plt.legend()
        plt.show()

    line_tmp=''
    for i in range(len(q_output_rest)):
        if i%3==0:#1st quantitity in the row
            line_tmp=line_tmp+('    %.15f'%q_output_rest[i])
        elif i%3==1:#2nd quantitity in the row
            line_tmp=line_tmp+('         %.15f'%q_output_rest[i])
        elif i%3==2:#3rd quantitity in the row
            line_tmp=line_tmp+('         %.15f      \n'%q_output_rest[i])
            file.write(line_tmp)
            line_tmp=''
    if len(q_output_rest)%3!=0:
        file.write(line_tmp+'      \n')


    #After the q 
    for i in range(end_index,len(sdata)):
        file.write(sdata[i]+'\n')



    



    