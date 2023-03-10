import os
import time
import shutil
#update the version

def text_to_dict(file_name):
    
    input_gacode_file=file_name
    f=open(input_gacode_file,"r")

    input_param={}
    lines=f.readlines()
    for line in lines:
        line=line.strip('\n\t ').replace(' ','')
        #print(line)
        if '#' in line[:2]:
            pass
        elif '=' in line:
            tmp=line.split('#')[0]
            #print('**************')
            #print(tmp)
            tmp=tmp.split('=')
            
            input_param[tmp[0]]=tmp[1]
            
    return input_param

def replace(inputFile,exportFile,target,replacement):
   inputFile = open(inputFile, 'r') 
   lines_original=inputFile.readlines()
   inputFile.close()
   
   exportFile = open(exportFile, 'w')
   for line in lines_original:
      new_line = line.replace(target, replacement)
      exportFile.write(new_line) 
   exportFile.close()


input_para = text_to_dict('setup.py')

version=input_para['VERSION']

version_num=version.split('.')
new_version_str=version[:len(version)-len(version_num[-1])]\
                +str(int(version_num[-1][:-1])+1)+"'"

if 1==1:
    print(input_para['VERSION'])
    print(new_version_str)



replace('setup.py','setup.py',\
         input_para['VERSION'],new_version_str)


#remove the distribution
shutil.rmtree('SLiM_phys.egg-info')
shutil.rmtree('dist')
shutil.rmtree('build')

#upload the package
os.system('python setup.py sdist bdist_wheel')
os.system('twine upload dist/*')


#update the package
#time.sleep(1)
os.system('pip install SLiM-phys -U')
print('run the following line to update')
print('pip install SLiM-phys -U')