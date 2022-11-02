import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Util():

    def __init__(self,args):
        self.args = args
        self.n_level = args.n_level
        self.scale = 0.5
        self.GPU =args.gpu

    def get_pyramid(self,img):
        img_pyramid=[]
        for i in range(self.n_level-1):
            scale_f = self.scale ** (self.n_level - i -1)
            down = nn.Upsample(scale_factor=scale_f, mode = 'bilinear',align_corners=True)
            img_pyramid.append(down(img))

        img_pyramid.append(img)
        return img_pyramid

    def multi_scale_loss(self,gt_pyramid,deblur_pyramid):
        mse = nn.MSELoss().cuda(self.GPU)
        loss = 0
        for i in range(self.n_level):
            loss += mse(deblur_pyramid[i],gt_pyramid[i])
        return loss

    def im2uint8(self,x):
        x = x.cpu().numpy()
        x= (x)*255
        out = np.clip(x,0,255)+0.5
        return out.astype(np.uint8)

    def gim2uint8(self,x):
        x = x.detach().cpu().numpy()
        x= (x)*255
        out = np.clip(x,0,255)+0.5
        return out.astype(np.uint8)
