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


def decompress_from_nnp(nnp_path):
    with zipfile.ZipFile(nnp_path,'r') as nnp:
        for name in nnp.namelist():
            nnp.extract(name,"./tmp")
    remove_glob('tmp/*.protobuf')



def compress_to_nnp(nnp_path,method):
    with zipfile.ZipFile(nnp_path+'test_'+method+'.nnp','w') as nnp:
        nnp.write('tmp/nnp_version.txt','nnp_version.txt')
        nnp.write('tmp/network.nntxt','network.nntxt')
        nnp.write('tmp/test_{}.protobuf'.format(method),'test_{}.protobuf'.format(method))

#def del_tmp_dir()


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument("-p","--path",type=str,help="path to the original nnp model",default="tmp.monitor/lenet_result.nnp")
    parser.add_argument("-t","--translate",type=str,default="ap2",help="choice the translating method from AP2,G1 or G2")
    args=parser.parse_args()

    os.makedirs("./tmp")

    decompress_from_nnp(args.path)

    nnp_copy="tmp.monitor/lenet_copy.nnp"
    shutil.copyfile(args.path,nnp_copy)

    nn.load_parameters(nnp_copy)
    param_dict=nn.parameter.get_parameters()
    my_param_dict=param_dict.copy()
    for key in param_dict.keys():
        param=nn.parameter.get_parameter(key)
        my_param=nn.Variable(param.shape)
        my_param.d=param.d
        #この条件文はfor文の外側にあるべき
        if(args.translate=="ap2"):
            #0に対してLogはとれないので対処しなければならないので無理やり2の-8乗を足している
            my_param.d=np.sign(param.d)*(2**(np.round(np.log2(np.abs(param.d)+2**(-8)))))
        elif(args.translate=="g1"):
            cor=np.zeros(param.shape)
            my_param.d=np.sign(param.d)*(2**(np.round(np.log2(np.abs(param.d)+2**(-8)))))
            cor=param.d-my_param.d
            cor=np.sign(cor)*(2**(np.round(np.log2(np.abs(cor)+2**(-8)))))
            my_param.d=my_param.d+cor

        # elif(args.translate=="g2"):
        else:
            raise ValueError("Unknown translate method {}".format(args.translate))
        nn.parameter.set_parameter(key,my_param)
    nn.parameter.save_parameters('./tmp/test_{}.protobuf'.format(args.translate))
    
    compress_to_nnp('tmp.monitor/',args.translate)
    
    remove_glob('tmp/*')
    os.rmdir("./tmp")
    os.remove("./tmp.monitor/lenet_copy.nnp")
