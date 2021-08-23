"""
torch中的Dataset定义脚本
"""
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader

def make_filelist():
    image_path = '/home/data/LiTS/fixed_data/data'
    mask_path = '/home/data/LiTS/fixed_data/label'

    image_save_dir = 'data/'
    mask_save_dir = 'label/'
    imagenames = os.listdir(image_path)
    masknames = os.listdir(mask_path)

    listfile = '../datatxt/LitsSeg.txt'
    fw = open(listfile,'w+')
    for imagename,maskname in zip(imagenames,masknames):
        image_save_path = os.path.join(image_save_dir,imagename)
        mask_save_path = os.path.join(mask_save_dir,imagename.replace("volume","segmentation"))
        fw.write(image_save_path+' '+mask_save_path +'\n')

class Lits_DataSet(dataset):

    def __init__(self,root,listfile,depth,augment=False):
        self.root = root 
        self.filenames = []
        self.ct_list = []
        self.seg_list = []
        self.depth = depth
        fr = open(listfile, 'r')
        lines = fr.readlines()
        for line in lines:
            linesplit = line.split(' ')
            image = os.path.join(root,linesplit[0])
            mask = os.path.join(root,linesplit[1])
            self.ct_list.append(image.strip())
            self.seg_list.append(mask.strip())
            self.filenames.append(linesplit[0])
    

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # min max 归一化
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 在slice平面内随机选取48张slice
        start_slice = random.randint(0, ct_array.shape[0] - self.depth)
        end_slice = start_slice + self.depth - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # 多任务，同时预测liver 和 tumor
        labelShape = seg_array.shape
        nplabel = np.empty((2, labelShape[0], labelShape[1], labelShape[2]))
        t1label = seg_array.copy()
        t1label[seg_array == 2] = 1 # 肝脏
    
        t2label = seg_array.copy()
        t2label[seg_array == 1] = 0 # 病灶
        t2label[seg_array == 2] = 1
        nplabel[0, :, :, :] = t1label
        nplabel[1, :, :, :] = t2label

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(nplabel)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)

if __name__ == "__main__":
    root = '/home/data/LiTS/fixed_data/'
    file_txt = '../datatxt/LitsSeg_train.txt'
    train_dataset = Lits_DataSet(root,file_txt,48) 
    dataloader = DataLoader(train_dataset)
    for (ct,seg) in dataloader:
        print(f'ct_array and seg_array: {ct.shape}, seg_array : {seg.shape}')
        # assert 1>3