import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import random
from models.model import SRNet
from trainer.train import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.datasets_dic import GoProDataset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='deblur arguments')
parser.add_argument("-e","--epochs",type = int, default = 2000)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 16)
parser.add_argument("-s","--cropsize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("--n_level",type=int, default=3)

parser.add_argument("--train_datalist",type=str, default='./datalist/datalist_gopro.txt')
parser.add_argument("--data_root_dir",type=str, default='./dataset')
parser.add_argument("--checkdir",type=str,default='./checkpoint/srnet')
parser.add_argument('--max_iteration', type=int, default=524000)
parser.add_argument("--isloadch", type=int,default=0)
parser.add_argument("--load_checkdir",type=str,default='./checkpoint/srnet')
parser.add_argument("--load_epoch",type=int,default=0)

parser.add_argument("--isval", action="store_true")
parser.add_argument("--mgpu", action="store_true")

parser.set_defaults(isval=False)
parser.set_defaults(mgpu=False)
args = parser.parse_args()

#Hyper Parameters
LEARNING_RATE = args.learning_rate
GPU = args.gpu
BATCH_SIZE = args.batchsize
CROP_SIZE = args.cropsize

##initial
data_root_dir = args.data_root_dir
load_epoch=args.load_epoch
train_log_dir = os.path.join(args.checkdir, 'tlog')
checkdir = args.checkdir


random.seed(777)
np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
torch.backends.cudnn.benchmark = False

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def main():
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    
    if not os.path.exists(checkdir):
        os.makedirs(checkdir)

    train_writer = SummaryWriter(logdir=train_log_dir)
    SRNet = SRNet()

    if args.isloadch: 
        load_path = os.path.join(args.load_checkdir,'Epoch%04d'%(load_epoch))
        if os.path.exists(str(load_path + "/SRNet.pkl")):
            SRNet.load_state_dict(torch.load(str(load_path + "/SRNet.pkl")))
            print("load SRNet success")
    else:
        print("weight initializing...")
        SRNet.apply(weight_init).cuda(GPU)

    if args.mgpu:
        SRNet = nn.DataParallel(SRNet)

    train_dataset = GoProDataset(
      image_list = args.train_datalist,
      root_dir = data_root_dir,
      crop = True,
      crop_size = CROP_SIZE,
      transform = transforms.Compose([
        #ToTensor transforms the image to a tensor with range [0,1]
        transforms.ToTensor()
    ]))

    train_dataloader = DataLoader(train_dataset, num_workers=3, batch_size = BATCH_SIZE, shuffle=True)
    optim = torch.optim.Adam(SRNet.parameters(),lr=LEARNING_RATE)
    mse = nn.MSELoss().cuda(GPU)
    Trainer(args).train(SRNet,train_dataloader,optim,mse,train_writer)

    train_writer.close()

if __name__ == '__main__':
    main()
