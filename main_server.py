import os
import sys
import torch
sys.path.append('/home.nfs/vicenluk/KSY/Illusory-Contour-Predictive-Networks/my_scripts/scripts/')
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
from classification import classification
from train_process import *
from train_params import *
from mydata import *
PATH = "/home.nfs/vicenluk/KSY/Illusory-Contour-Predictive-Networks/"

trainroot = 'train.txt'
validroot = 'valid.txt'
batch = 40
#pretraining(device='cuda:2',dataroot=PATH,trainbatch=512,valbatch=512)
checkpoint = torch.load('train/train1_t10_net1_ep191.pth') 
finetuning(device='cuda:2',alpha=0.1,train_root = trainroot,valid_root=validroot,root_path=PATH+'finetuned',trainbatch=batch,valbatch=40, checkpoint1 = checkpoint)

#batch      = 40
#saveroot   = f'./results/prob_net11.pth'
#dataroot   = './data/illusory0.1/test.txt'
#checkpoint = torch.load(f'./train/train2_net11_ep24.pth') # load the finetuned netowrk

#classification(device,checkpoint,dataroot,batch,saveroot)

#from scripts.plotting import plotting_class

#img_num=1200  # the number of each testing class
#saveflag = True
#prob_square = torch.load(f'results/prob_net11.pth') 
#saveroot = f'pic/decision_net11.pdf'

#plotting_class(prob_square,saveflag,saveroot,img_num)


