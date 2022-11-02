import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.model import SRNet
import os
import argparse
import time
import imageio

parser = argparse.ArgumentParser(description='SRN pytorch test')
parser.add_argument("--n_level",type=int, default=3)
parser.add_argument("--test_datalist",type=str, default='./datalist/datalist_gopro_testset.txt')
parser.add_argument("--data_root_dir",type=str,default='./dataset')
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument("-outdir",type=str,default='./result/srnet')

parser.add_argument("-write_time", action="store_true")
args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

data_root_dir = args.data_root_dir
GPU = args.gpu
n_level = args.n_level
scale = 0.5
outdir = args.outdir
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

if args.write_time:
    time_measure_dir = os.path.join(args.outdir,'measure')
    if not os.path.exists(time_measure_dir):
        os.makedirs(time_measure_dir)

    time_txt = open(os.path.join(time_measure_dir,'srn_py_gopro_time.txt'),'w')


def deblur_processe(blur_image,h,w,model):
    inp_pred = blur_image
    for i in range(n_level):
        scale_f = scale ** (n_level - i -1)
        hi = int(round(h * scale_f))
        wi = int(round(w * scale_f))
        interpolation = nn.Upsample(size=[hi,wi], mode = 'bilinear',align_corners=True)
        inp_blur = interpolation(blur_image)
        up_inp_pred = interpolation(inp_pred)

        inp_all = torch.cat([inp_blur,up_inp_pred],dim=1)#concat blur_pyramid[i], inp_pred
        inp_pred = model(inp_all)

    deblur_image = inp_pred
    return deblur_image


def main():
    test_data_list = open(args.test_datalist,'rt').read().splitlines()
    test_data_list = list(map(lambda x: x.strip().split(' '), test_data_list))
    test_num = len(test_data_list)
    print('test num: ',test_num)
    model = SRNet().cuda(GPU)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total param:%d, trainable total param:%d'%(pytorch_total_params,pytorch_total_params_train))
    model = nn.DataParallel(model)

    model.eval()
    with torch.no_grad():

        for itr, data in enumerate(test_data_list):
            blur_image_dir = os.path.join(data_root_dir, data[1])
            blur = imageio.imread(blur_image_dir)

            blur_img = blur.astype('float32')

            h = int(blur_img.shape[0])
            w = int(blur_img.shape[1])

            blur_img = blur_img / 255

            blur_image = transforms.ToTensor()(blur_img)
            blur_image = blur_image.unsqueeze(0).cuda(GPU)

            _ =deblur_processe(blur_image,h,w,model)

            if itr == 10:
                break

        iteration = 0.0
        itr_time = 0.0
        for _, blur_path in test_data_list:
            iteration += 1
            blur_image_dir = os.path.join(data_root_dir, blur_path)
            blur = imageio.imread(blur_image_dir)

            blur_img = blur.astype('float32')

            h = int(blur_img.shape[0])
            w = int(blur_img.shape[1])

            blur_img = blur_img / 255

            blur_image = transforms.ToTensor()(blur_img)
            blur_image = blur_image.unsqueeze(0).cuda(GPU)

            torch.cuda.synchronize()
            init_time = time.time()
            _=deblur_processe(blur_image,h,w,model)
            torch.cuda.synchronize()
            cur_time = time.time() - init_time
            itr_time += cur_time
            print('%04dth img, time:%f'%(iteration,cur_time))

    average_time = itr_time/test_num
    print('average test time: %f'%(average_time))
    if args.write_time:
        time_txt.write('srn py gopro average time: %.4f'%(average_time))
        time_txt.close()


if __name__ == '__main__':
    main()
