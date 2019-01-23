import nnabla as nn
import numpy as np
import os
import shutil
import zipfile
import glob
import argparse

def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)

os.makedirs("./tmp",exist_ok=True)
with zipfile.ZipFile("tmp.monitor/lenet_result.nnp",'r') as nnp:
    for name in nnp.namelist():
        nnp.extract(name,"./tmp")
remove_glob('tmp/*.protobuf')

if(os.path.exists("tmp.monitor/lenet_copy.nnp")==False):
    shutil.copyfile("tmp.monitor/lenet_result.nnp","tmp.monitor/lenet_copy.nnp")
nn.load_parameters("tmp.monitor/lenet_copy.nnp")

param_dict=nn.parameter.get_parameters()

for key in param_dict.keys():
    param=nn.parameter.get_parameter(key)
    #0に対してLogはとれないので対処しなければならないので無理やり2の-8乗を足している
    param.d=np.sign(param.d)*(2**(np.round(np.log2(np.abs(param.d)+2**(-8)))))
    nn.parameter.set_parameter(key,param)

nn.parameter.save_parameters('./tmp/test.protobuf')

with zipfile.ZipFile('tmp.monitor/test.nnp','w') as nnp:
    nnp.write('tmp/nnp_version.txt','nnp_version.txt')
    nnp.write('tmp/network.nntxt','network.nntxt')
    nnp.write('tmp/test.protobuf','test.protobuf')



    
