from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import Dataset
import random
import os
import OpenEXR
import logging
import numpy
import Imath
from springvision import Normalize, ToTensor, Compose
logger = logging.getLogger('global')
import re
import numpy as np
import sys
import yaml 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    #print(header)
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', str(file.readline(),encoding='utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale



transforms_info_dict = {
    'normalize': Normalize,
    'to_tensor': ToTensor,
    'compose': Compose,
}
def build_transformer():
    cfgs = yaml.load(open(os.path.join(os.getcwd(),'./dataset/cfg.yaml')), Loader=yaml.FullLoader)
    transform_list = []
    for _, cfg in cfgs.items():
        transform_type = transforms_info_dict[cfg['type']]
        kwargs = cfg['kwargs'] if 'kwargs' in cfg else {}
        transform = transform_type(**kwargs)
        transform_list.append(transform)
    return transforms_info_dict['compose'](transform_list)

def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
    	Channels = ['R', 'G', 'B']
    	CNum = 3
    else:
    	Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in Channels]
    hdr = numpy.zeros((Size[1],Size[0],CNum),dtype=numpy.float32)
    if (CNum == 1):
        hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
    else:
	    hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
	    hdr[:,:,1] = numpy.reshape(Pixels[1],(Size[1],Size[0]))
	    hdr[:,:,2] = numpy.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def writehdr(hdrpath,hdr):
	h, w, c = hdr.shape
	if c == 1:
		hdr = numpy.pad(hdr, ((0, 0), (0, 0), (0, 2)), 'constant')
		hdr[:,:,1] = hdr[:,:,0]
		hdr[:,:,2] = hdr[:,:,0]
	imageio.imwrite(hdrpath,hdr,format='hdr')

def load_exr(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = numpy.squeeze(hdr)
	return hdr
    
def load_rgb(filename):
    img = None
    if filename.find('.npy') > 0:
        img = np.load(filename)
    else:
        img = Image.open(filename).convert('RGB')
    return img


def load_disp(filename):
    gt_disp = None
    if filename.endswith('pfm'):
        gt_disp, scale = readPFM(filename)
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32)
    elif filename.endswith('npy'):
        gt_disp = np.load(filename)
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32)
    elif filename.endswith('exr'):
        gt_disp = load_exr(filename)
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32)
    elif filename.find('kitti') > -1:
        gt_disp = Image.open(filename)
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32) / 256
    elif filename.find('InStereo2K') > -1:
        gt_disp = Image.open(filename)
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32) / 100
    elif filename.endswith('png'):
        gt_disp = Image.open(filename)
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32) / 256

    return gt_disp


class FileDataset(Dataset):
    def __init__(self, root_dir, listfile, mode='train', height=None, width=None, augment=False, stride=64, **kwargs):
        self.mode = mode
        self.height = height
        self.width = width
        self.augment = augment
        self.stride = stride

        fr = open(listfile, 'r')
        lines = fr.readlines()
        self.left_paths = []
        self.right_paths = []
        self.left_disp_paths = []
        self.right_disp_paths = []
        self.filenames = []
        for line in lines:
            linesplit = line.split(' ')
            leftpath = os.path.join(root_dir, linesplit[0])
            rightpath = os.path.join(root_dir, linesplit[1])

            self.left_paths.append(leftpath.strip())
            self.right_paths.append(rightpath.strip())
            self.filenames.append(linesplit[0].strip())

            if len(linesplit) > 2:
                disppath = os.path.join(root_dir, linesplit[2])
                self.left_disp_paths.append(disppath.strip())

            if len(linesplit) > 3:
                disppath = os.path.join(root_dir, linesplit[3])
                self.right_disp_paths.append(disppath.strip())

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = load_rgb(self.left_paths[idx])
        right_img = load_rgb(self.right_paths[idx])
        file_name = self.filenames[idx]
        w, h = left_img.size

        left_disp = None
        right_disp = None

        if self.mode == 'train':
            tw = self.width
            th = self.height

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            left_disp = load_disp(self.left_disp_paths[idx])
            left_disp = left_disp[y1:y1 + th, x1:x1 + tw]

            if len(self.right_disp_paths) != 0 and os.path.isfile(self.right_disp_paths[idx]):
                right_disp = load_disp(self.right_disp_paths[idx])
                right_disp = right_disp[y1:y1 + th, x1:x1 + tw]

        elif self.mode == 'val':
            tw = w // self.stride * self.stride
            th = h // self.stride * self.stride
            
            left_img = left_img.crop((w-tw, h-th, w, h))
            right_img = right_img.crop((w-tw, h-th, w, h))

            left_disp = load_disp(self.left_disp_paths[idx])
            left_disp = left_disp[h-th:h, w-tw:w]

        elif self.mode == 'test':
            if self.width is None:
                new_w = ((w-1)//self.stride + 1) * self.stride
                new_h = ((h-1)//self.stride + 1) * self.stride
                bottom_pad = new_h - h
                right_pad = new_w - w

                left_img = Image.fromarray(np.pad(left_img, ((0, bottom_pad), (0, right_pad), (0, 0)), mode='constant', constant_values=0))
                right_img = Image.fromarray(np.pad(right_img, ((0, bottom_pad), (0, right_pad), (0, 0)), mode='constant', constant_values=0))
            else:
                left_img = left_img.resize((self.width,self.height), Image.BILINEAR)
                right_img = right_img.resize((self.width, self.height), Image.BILINEAR)

            if len(self.left_disp_paths) != 0 and os.path.isfile(self.left_disp_paths[idx]):
                left_disp = load_disp(self.left_disp_paths[idx])

            if len(self.right_disp_paths) != 0 and os.path.isfile(self.right_disp_paths[idx]):
                right_disp = load_disp(self.right_disp_paths[idx])
        
        data = {'left_img': left_img,
                'right_img': right_img}
        if left_disp is not None:
            data['left_disp'] = left_disp
        if right_disp is not None:
            data['right_disp'] = right_disp
        # transformer = preprocess.get_transform(augment=self.augment)
        transformer = build_transformer()
        data['left_img'] = transformer(data['left_img'])
        data['right_img'] = transformer(data['right_img'])
        data['left_disp'] = transformer(data['left_disp'])
        data['file_name'] = file_name
        data['origin_h'] = h
        data['origin_w'] = w

        return data
