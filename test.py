import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import model
import os
import math
import argparse
import imageio

parser = argparse.ArgumentParser(description='SRN pytorch test')
parser.add_argument("--n_level",type=int, default=3)
parser.add_argument("--test_datalist",type=str, default='./datalist/datalist_gopro_test.txt')
parser.add_argument("--deblur_datalist",type=str, default='./datalist/pytorch_gopro_eval.txt')
parser.add_argument("--load_dir",type=str,default='./checkpoint/Epoch4000')
parser.add_argument("--outdir",type=str,default='./result/srnet')
parser.add_argument("--data_root_dir",type=str,default='./dataset')
parser.add_argument("--gpu",type=int, default=0)

args = parser.parse_args()

data_root_dir = args.data_root_dir
GPU = args.gpu
outdir = args.outdir
n_level = args.n_level
scale = 0.5

if not os.path.exists(outdir):
    os.makedirs(outdir)

def deblur_processe(blur_image,h,w,SRNet):
    inp_pred = blur_image
    for i in range(n_level):
        scale_f = scale ** (n_level - i -1)
        hi = int(round(h * scale_f))
        wi = int(round(w * scale_f))
        interpolation = nn.Upsample(size=[hi,wi], mode = 'bilinear',align_corners=True)
        inp_blur = interpolation(blur_image)
        up_inp_pred = interpolation(inp_pred)

        inp_all = torch.cat([inp_blur,up_inp_pred],dim=1)
        inp_pred = SRNet(inp_all)

    deblur_image = inp_pred
    return deblur_image


def main():
    pytorch_total_params = sum(p.numel() for p in SRNet.parameters())
    pytorch_total_params_train = sum(p.numel() for p in SRNet.parameters() if p.requires_grad)
    print('total param:%d, trainable total param:%d'%(pytorch_total_params,pytorch_total_params_train))

    test_data_list = open(args.test_datalist,'rt').read().splitlines()
    test_data_list = list(map(lambda x: x.strip().split(' '), test_data_list))
    deblur_txt = open(args.deblur_datalist,'w')

    print('test num: ',len(test_data_list))
    SRNet = model.SRNet().cuda(GPU)
    checkdir = str(args.load_dir + "/SRNet.pkl")
    if os.path.exists(checkdir):
        SRNet.load_state_dict(torch.load(checkdir))
        print("load SRNet success")
    else:
        print("Checkpoint doesn't exist")
        raise ValueError

    SRNet.eval()
    with torch.no_grad():
        iteration = 0
        for gt_path,blur_path in test_data_list:
            blur = imageio.imread(os.path.join(data_root_dir, blur_path))
            gt = imageio.imread(os.path.join(data_root_dir, gt_path))

            blur_img = blur.astype('float32')

            h = int(blur_img.shape[0])
            w = int(blur_img.shape[1])

            per = math.pow(2,7)

            if (h % per) == 0:
                new_h = h
            else:
                new_h = h - (h % per)
            if (w % per) == 0:
                new_w = w
            else:
                new_w = w - (w % per)
            print(new_h,new_w)
            blur_img_crop = blur_img[:int(new_h),:int(new_w),:]

            blur_img_crop = blur_img_crop / 255

            blur_image = transforms.ToTensor()(blur_img_crop)
            blur_image = blur_image.unsquimg_nameeeze(0).cuda(GPU)

            deblur_image=deblur_processe(blur_image,new_h,new_w,SRNet)

            out = deblur_image.data
            if (new_h - h) > 0 or (new_w - w) > 0:
                out = out[:, :, :h, :w]

            out = out[0,:,:,:].cpu().numpy()
            del blur_image
            del deblur_image
            torch.cuda.empty_cache()
            out = np.transpose(out,(1,2,0))

            out = np.clip(out * 255, 0, 255) + 0.5
            out = out.astype('uint8')

            split_blur_path = blur_path.split('/')
            img_name = split_blur_path[-1].split('.')[0]
            deblur_name = img_name +'_deblur.png'
            gt_name = img_name +'gt.png'
            
            deblur_path = outdir + '/' + split_blur_path[1] + '/' + "deblur"
            crop_gt_path = outdir + '/' + split_blur_path[1] + '/' + "gt"

            if not os.path.exists(deblur_path):
                os.makedirs(deblur_path)

            if not os.path.exists(crop_gt_path):
                os.makedirs(crop_gt_path)

            deblur_img_dir = deblur_path + '/' + deblur_name
            gt_img_dir = crop_gt_path + '/' + gt_name

            print('iter :',iteration)
            gt_img_crop = gt[:int(new_h),:int(new_w),:]
            imageio.imwrite(deblur_img_dir, out)
            imageio.imwrite(gt_img_dir, gt_img_crop)

            deblur_txt.write(gt_img_dir+' '+deblur_img_dir+'\n')

            iteration += 1
    print('Test Finish')

if __name__ == '__main__':
    main()
