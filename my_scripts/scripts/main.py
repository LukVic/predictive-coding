from train_process import *
from train_params import *
from mydata import *

PATH = '/media/lucas/ADATA SE760/KYR/KSY/Illusory-Contour-Predictive-Networks/my_scripts/data'

def main():
    pretraining(device='cuda',dataroot=PATH,trainbatch=128,valbatch=128)
    

if __name__ == '__main__':
    main()