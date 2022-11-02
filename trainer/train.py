import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from util import Util
from torch_poly_lr_decay import PolynomialLRDecay
import os

class Trainer():
    def __init__(self,args):
        self.arg=args
        self.n_level = args.n_level
        self.scale = 0.5
        self.loadepoch=args.loadepoch
        self.EPOCHS = args.epochs
        self.GPU =args.gpu
        self.CROP_SIZE = args.cropsize
        self.mgpu = args.mgpu

    def train(self,SRNet,train_dataloader,optim,mse,train_writer):
        if self.arg.isloadch:
            start_epoch = self.loadepoch+1
        else:
            start_epoch = 1

        train_batch_num = len(train_dataloader)
        max_step = train_batch_num * self.EPOCHS
        print('steps:',max_step)
        scheduler = PolynomialLRDecay(optim, max_decay_steps=max_step, end_learning_rate=0.0, power=0.3)

        all_iteration=0
        for epoch in range(start_epoch,self.EPOCHS+1):
            print("epoch: ",epoch)
            sum_trainloss = 0
            start = 0
            for iteration, images in enumerate(train_dataloader):
                train_writer.add_scalar('lr',optim.param_groups[0]['lr'], all_iteration)
                all_iteration+=1

                gt = Variable(images['sharp_image']).cuda(self.GPU)
                blur_images = Variable(images['blur_image']).cuda(self.GPU)

                inp_pred = blur_images
                loss=0

                for i in range(self.n_level):
                    scale_f = self.scale ** (self.n_level - i -1)
                    hi = int(round(self.CROP_SIZE * scale_f))
                    wi = int(round(self.CROP_SIZE * scale_f))
                    interpolation = nn.Upsample(size=[hi,wi], mode = 'bilinear',align_corners=True)
                    inp_blur = interpolation(blur_images)
                    inp_pred  = interpolation(inp_pred).detach()
                    inp_gt = interpolation(gt)

                    inp_all = torch.cat([inp_blur,inp_pred],dim=1)
                    inp_pred = SRNet(inp_all)
                    loss += mse(inp_gt,inp_pred)

                SRNet.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                sum_trainloss += loss.item()

                if (iteration+1)%10 == 0:
                    stop = time.time()
                    print("epoch:", epoch, "iteration:", iteration+1, "loss:%.4f"%loss.item(),
                    '(%.3f s/10iter)'%(stop-start))
                    start = time.time()

            train_writer.add_scalar('training loss',sum_trainloss/train_batch_num, epoch)
            train_writer.add_images('deblur images',Util(self.arg).gim2uint8(inp_pred),epoch)
            train_writer.add_images('blur images',Util(self.arg).im2uint8(blur_images),epoch)
            train_writer.add_images('gt images',Util(self.arg).im2uint8(gt),epoch)

            if epoch==1 or epoch%100 == 0 or epoch == 2000:
                print('Saving...')
                checkpath = self.arg.checkdir  + '/Epoch%04d'%(epoch)
                if os.path.exists(checkpath) == False:
                    os.makedirs(checkpath)

                if self.mgpu:
                    torch.save(SRNet.module.state_dict(),str(checkpath+ "/SRNet.pkl"))
                else:
                    torch.save(SRNet.state_dict(),str(checkpath+ "/SRNet.pkl"))
