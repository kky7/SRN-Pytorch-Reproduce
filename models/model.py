import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/pytorch/pytorch/issues/3867

def Conv(in_feats,out_feats,kernel_size,stride,pad_size):
    return nn.Conv2d(in_feats,out_feats,kernel_size=kernel_size,stride=stride,padding=pad_size)

def TransConv(in_feats,out_feats,kernel_size,stride,pad_size): 
    return nn.ConvTranspose2d(in_feats,out_feats, kernel_size=kernel_size, stride=stride, padding=pad_size)

class ResBlock(nn.Module):
    def __init__(self,n_feats,kernel_size,pad_size=2):
        super(ResBlock, self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(n_feats,n_feats,kernel_size=kernel_size,stride=1,padding=pad_size),
            nn.ReLU(),
            nn.Conv2d(n_feats,n_feats,kernel_size=kernel_size,stride=1,padding=pad_size)
        )

    def forward(self,x):
        x = self.block(x) + x
        return x

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet,self).__init__()
        self.relu = nn.ReLU()
        #scale weight sharing (scale recurrent)
        self.enc1 = Conv(6,32,kernel_size=5,stride=1,pad_size=2)
        self.Eres1_1 = ResBlock(32,kernel_size=5,pad_size=2)
        self.Eres1_2 = ResBlock(32,kernel_size=5,pad_size=2)
        self.Eres1_3 = ResBlock(32,kernel_size=5,pad_size=2)

        self.enc2 = Conv(32,64,kernel_size=5,stride=2,pad_size=2)
        self.Eres2_1 = ResBlock(64,kernel_size=5,pad_size=2)
        self.Eres2_2 = ResBlock(64,kernel_size=5,pad_size=2)
        self.Eres2_3 = ResBlock(64,kernel_size=5,pad_size=2)

        self.enc3 = Conv(64,128,kernel_size=5,stride=2,pad_size=2)
        self.Eres3_1 = ResBlock(128,kernel_size=5,pad_size=2)
        self.Eres3_2 = ResBlock(128,kernel_size=5,pad_size=2)
        self.Eres3_3 = ResBlock(128,kernel_size=5,pad_size=2)

        self.Dres1_1 = ResBlock(128,kernel_size=5,pad_size=2)
        self.Dres1_2 = ResBlock(128,kernel_size=5,pad_size=2)
        self.Dres1_3 = ResBlock(128,kernel_size=5,pad_size=2)
        self.dec1 = TransConv(128,64,kernel_size=4,stride=2,pad_size=1)

        self.Dres2_1 = ResBlock(64,kernel_size=5,pad_size=2)
        self.Dres2_2 = ResBlock(64,kernel_size=5,pad_size=2)
        self.Dres2_3 = ResBlock(64,kernel_size=5,pad_size=2)
        self.dec2 = TransConv(64,32,kernel_size=4,stride=2,pad_size=1)

        self.Dres3_1 = ResBlock(32,kernel_size=5,pad_size=2)
        self.Dres3_2 = ResBlock(32,kernel_size=5,pad_size=2)
        self.Dres3_3 = ResBlock(32,kernel_size=5,pad_size=2)
        self.dec3= Conv(32,3,kernel_size=5,stride=1,pad_size=2)


    def forward(self,x):
        conv1_1 = self.enc1(x)
        conv1_1 = self.relu(conv1_1)
        conv1_2 = self.Eres1_1(conv1_1)
        conv1_3 = self.Eres1_2(conv1_2)
        conv1_4 = self.Eres1_3(conv1_3)

        conv2_1 = self.enc2(conv1_4)
        conv2_1 = self.relu(conv2_1)
        conv2_2 = self.Eres2_1(conv2_1)
        conv2_3 = self.Eres2_2(conv2_2)
        conv2_4 = self.Eres2_3(conv2_3)

        conv3_1 = self.enc3(conv2_4)
        conv3_1 = self.relu(conv3_1)
        conv3_2 = self.Eres3_1(conv3_1)
        conv3_3 = self.Eres3_2(conv3_2)
        conv3_4 = self.Eres3_3(conv3_3)

        deconv1_1 = self.Dres1_1(conv3_4)
        deconv1_2 = self.Dres1_2(deconv1_1)
        deconv1_3 = self.Dres1_3(deconv1_2)
        deconv1_4 = self.dec1(deconv1_3)
        deconv1_4 = self.relu(deconv1_4)

        skip1 = deconv1_4 + conv2_4
        deconv2_1 = self.Dres2_1(skip1)
        deconv2_2 = self.Dres2_2(deconv2_1)
        deconv2_3 = self.Dres2_3(deconv2_2)
        deconv2_4 = self.dec2(deconv2_3)
        deconv2_4 = self.relu(deconv2_4)

        skip2 = deconv2_4 + conv1_4
        deconv3_1 = self.Dres3_1(skip2)
        deconv3_2 = self.Dres3_2(deconv3_1)
        deconv3_3 = self.Dres3_3(deconv3_2)
        out = self.dec3(deconv3_3)

        return out
