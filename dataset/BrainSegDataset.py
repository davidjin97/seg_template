import numpy as np 
import torch
import torch.nn as nn 
import os 
from torch.utils.data import Dataset

def make_filelist():
    image_path = '/home/data/BraTs_Seg_Study/data/trainImage'
    mask_path = '/home/data/BraTs_Seg_Study/data/trainMask'

    image_save_dir = 'trainImage/'
    mask_save_dir = 'trainMask/'
    imagenames = os.listdir(image_path)
    masknames = os.listdir(mask_path)

    listfile = './BrainSeg.txt'
    fw = open(listfile,'w+')
    for imagename,maskname in zip(imagenames,masknames):
        image_save_path = os.path.join(image_save_dir,imagename)
        mask_save_path = os.path.join(mask_save_dir,maskname)
        fw.write(image_save_path+' '+mask_save_path +'\n')


class BrainSegDataset(Dataset):
    def __init__(self,root,listfile,model='train',augment=False):
        self.root = root 
        self.filenames = []
        self.images = []
        self.masks = []
        fr = open(listfile, 'r')
        lines = fr.readlines()
        for line in lines:
            linesplit = line.split(' ')
            image = os.path.join(root,linesplit[0])
            mask = os.path.join(root,linesplit[1])
            self.images.append(image.strip())
            self.masks.append(mask.strip())
            self.filenames.append(linesplit[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = np.load(image)
        mask = np.load(mask)
        image = image.transpose(2,0,1).astype(np.float32)
        mask = mask[np.newaxis,:,:].astype(np.int64)
        print('mask: ',mask.dtype,idx)
        return image,mask

