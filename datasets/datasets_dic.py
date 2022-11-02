from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class GoProDataset(Dataset):
    def __init__(self,image_list, root_dir, crop=True, crop_size=256,transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        datalist = open(image_list, 'r')
        self.image_files = datalist.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.imgdic={}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        images_dir = self.image_files[idx][0:-1]
        split_images_dir = images_dir.split(' ')
        sharp_image_dir = split_images_dir[0]
        blur_image_dir = split_images_dir[1]

        final_blurdir = os.path.join(self.root_dir, blur_image_dir)
        final_sharpdir = os.path.join(self.root_dir, sharp_image_dir)

        if  blur_image_dir in self.imgdic:
            blur_image = self.imgdic[blur_image_dir]
            sharp_image = self.imgdic[sharp_image_dir]
        else:
            blur_image = Image.open(final_blurdir).convert('RGB')
            sharp_image = Image.open(final_sharpdir).convert('RGB')
            self.imgdic[blur_image_dir] = blur_image
            self.imgdic[sharp_image_dir] = sharp_image

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2]

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]

            blur_image = blur_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            sharp_image = sharp_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]

        return {'blur_image': blur_image, 'sharp_image': sharp_image}
