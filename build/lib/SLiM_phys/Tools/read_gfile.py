def read_gfile_to_dict(g_file_name):
    f=open(g_file_name,'r')
    data = f.read()
    f.close()

    sdata = data.split('\n')

    date = sdata[0].split()[1]
    discharge_num = int(sdata[0].split()[2][1:])
    time = int(sdata[0].split()[3][:-2])
    n0 = int(sdata[0].split()[4])
    nx = int(sdata[0].split()[5])
    ny = int(sdata[0].split()[6]) 

    print()

    g_quant_dict['q']={'header':''}

    return g_quant_dict
