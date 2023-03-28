import wget #pip install wget
import os.path

def download(download_path='./NN_model'):
    suffixes=['','_gamma','_omega','_stability']
    git_dir='NN_model'
    #download csv
    if os.path.exists(download_path):
        pass
    else:
        os.mkdir(download_path)
    for suffix in suffixes:
        input_file_name='NN'+suffix+'_norm_factor.csv'

        if os.path.isfile(download_path+'/'+input_file_name) :
            pass
        else:
            url="https://github.com/maxtcurie/SLiM/blob/main/"\
                    +git_dir+"/"+input_file_name+"?raw=true"
            wget.download(url, download_path+'/'+input_file_name)

    #download weight
    for suffix in suffixes:
        input_file_name='SLiM_NN'+suffix+'.h5'
        if os.path.isfile(download_path+'/'+input_file_name) :
            pass
        else:
            url="https://github.com/maxtcurie/SLiM/blob/main/"\
                    +git_dir+"/"+input_file_name+"?raw=true"
            wget.download(url, download_path+'/'+input_file_name)

if __name__ == "__main__":
    download(download_path='./NN_model')