import os
import sys
import torch
sys.path.append('/media/lucas/ADATA SE760/KYR/KSY/Illusory-Contour-Predictive-Networks/my_scripts/scripts/')
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
sys.path.append('my_scripts') # change the root accordingly
from classification import classification
from train_process import *
from train_params import *
from mydata import *
from plotting import plotting_class
from fgvalue import fg_compute


PATH = '/media/lucas/ADATA SE760/KYR/KSY/Illusory-Contour-Predictive-Networks/'
TRAIN_DATA = "train_data"
VALID_DATA = "valid_data"

batch      = 20
saveroot   = f'results/prob_net11.pth'
dataroot   = 'test.txt'
checkpoint = torch.load(f'./train/train2_net11_ep24.pth') # load the finetuned netowrk
trainroot = 'train.txt'
validroot = 'valid.txt'
test_root = 'test.txt'
batch_size = 20
#classification(device,checkpoint,dataroot,batch,saveroot)
#def finetuning(device,alpha,train_root,valid_root,root_path,trainbatch,valbatch,timesteps=10,net1number = 1,net2number=11,
    # max_epoch = 25,CheckpointFlag1 = False,checkpoint1 = None,checkpoint2=None):
#finetuning(device='cuda',alpha=0.1,train_root = trainroot,valid_root=validroot,root_path=PATH+'finetuned',trainbatch=batch,valbatch=40)

classification(device,checkpoint,dataroot,batch,saveroot)


img_num=500  # the number of each testing class
saveflag = True
prob_cube = torch.load(f'results/prob_net11.pth') 
saveroot = f'pic/decision_net11.pdf'

plotting_class(prob_cube,saveflag,saveroot,img_num)

#savepath = f'results/FG_net11.pth'
#fg_compute(device,checkpoint,test_root,batch_size,img_num,savepath)  