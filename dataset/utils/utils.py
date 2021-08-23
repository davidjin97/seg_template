import sys
import numpy as np
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch
import zipfile 
import os
import random

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message) #print to screen
        self.log.write(message) #print to logfile

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def find_bb(volume):
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 3
    # axis
    for i in range(img_shape[0]): ## z轴上的
        img_slice_begin = volume[i,:,:]
        if np.sum(img_slice_begin)>0:
            bb[0] = np.max([i-bb_extend, 0]) # 往前扩3帧
            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0]-1-i,:,:]
        if np.sum(img_slice_end)>0:
            bb[1] = np.min([img_shape[0]-1-i + bb_extend, img_shape[0]-1]) # 往后扩三帧
            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:,i,:]
        if np.sum(img_slice_begin)>0:
            bb[2] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:,img_shape[1]-1-i,:]
        if np.sum(img_slice_end)>0:
            bb[3] = np.min([img_shape[1]-1-i + bb_extend, img_shape[1]-1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:,:,i]
        if np.sum(img_slice_begin)>0:
            bb[4] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:,:,img_shape[2]-1-i]
        if np.sum(img_slice_end)>0:
            bb[5] = np.min([img_shape[2]-1-i+bb_extend, img_shape[2]-1])
            break
	
    return bb

def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image



def sitk_read_raw(img_path, resize_scale=1): # 读取3D图像并rescale（因为一般医学图像并不是标准的[1,1,1]scale）
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first
    nda=ndimage.zoom(nda,[resize_scale,resize_scale,resize_scale],order=0) #rescale
    return nda

# target one-hot编码
def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot

def make_one_hot_3d(x, n): # 对输入的volume数据x，对每个像素值进行one-hot编码
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n]) # 创建one-hot编码后shape的zero张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[i, j, v, int(x[i, j, v])] = 1 # 给相应类别的位置置位1，模型预测结果也应该是这个shape
    return one_hot

def random_crop_2d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]

    if random_x_max < 0 or random_y_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]

    return crop_img, crop_label

def random_crop_3d(img, label, crop_size): # crop_size = dhw 中心裁剪
    c,x,y = img.shape  # hwd
    startx = x//2 - crop_size[1]//2
    starty = y//2 - crop_size[2]//2   
    startc = c//2 - crop_size[0]//2 
    return img[startc:startc+crop_size[0], startx:startx+crop_size[1],starty:starty+crop_size[2]],\
        label[startc:startc+crop_size[0], startx:startx+crop_size[1],starty:starty+crop_size[2]]


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as f:
        file_name_list = f.readlines()
    return file_name_list

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr