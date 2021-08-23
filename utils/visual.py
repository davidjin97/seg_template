import cv2
import torch
import skimage.io
import numpy as np 
import os
from matplotlib import pyplot as plt


def visualization_from_single_channel(data,is_save=True,savename='./temp_vis_disp.png'):

    """visualization accord to a Tensor or Array whose shape is [1,1,h,w] or [1,h,w]
    Args:
        data: Tensor or Array 
        is_save: bool
        savename: string
    
    Returns:
        None or color data whose shape is [h,w,3]

    """
    if isinstance(data,torch.Tensor) :
        if data.is_cuda:
            data = data.detach().cpu().numpy()
        else:
            data = np.array(data)
    if len(data.shape)==4 and data.shape[0] == 1:
        data = data.squeeze(0)
    data = data.transpose(1,2,0)
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype('uint8')
    data = cv2.applyColorMap(data, cv2.COLORMAP_JET)

    if is_save:
        cv2.imwrite(savename,data)
        # skimage.io.imsave(savename, data)
    else:
        return data


def get_depth_from_disparity(FB,disp):

    """get depth from disparity
    Args:
        FB:  Focal lenth, Baseline
        disp: Tensor or Array

    Returns:
        depth: Tensor or Array

    """
    if isinstance(disp,torch.Tensor) :
        if disp.is_cuda:
            disp = disp.detach().cpu().numpy()
        else:
            disp = np.array(disp)
    if len(disp.shape)==4 and disp.shape[0] == 1:
        disp = disp.squeeze(0)
    disp = disp.transpose(1,2,0)
    return  fb / disp


def show_one_image_by_matplotlib(image):

    """ show image by matplotlib
    Args:
        image: Tensor or Array or PILImage
        
    Examples:
        filename = './left.png'
        image = Image.open(filename)
        show_image_by_matplotlib(image)
    """
    plt.figure()
    plt.title("Matplotlib image show") 
    plt.imshow(image) 
    plt.show()

def show_multi_image_by_matplotlib(images,rows,colums):
    """ show images by matplotlib
    Args:
        images: list and every item is Tensor or Array or PILImage
        rows: int
        colums: int
    Examples:
        images = []
        filename = './data/20200915214148920_image1.yuv_disp_color.png'
        images.append(Image.open(filename))
        filename = './data/20200915214148920_image1.yuv_disp_color_ins.png left.png'
        images.append(Image.open(filename))
        show_multi_image_by_matplotlib(images,1,2)
    """
    plt.figure()
    for row in range(rows):
        for col in range(colums):
            plt.subplot(rows,colums,row*colums+col+1)
            plt.imshow(images[row*colums+col])
    plt.show()
    
if __name__ == "__main__":
    pass 
