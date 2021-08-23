import argparse
import json
import os
import os.path as osp
import warnings
import PIL.Image
import yaml 
import base64
import numpy as np 


def get_depth_by_labelme(json_file,disp_file):

    depth = np.load(depth_file)
    result = []
    label_names = []
    if os.path.isfile(json_file):
        data = json.load(open(json_file))
        height = data['imageHeight']
        width = data['imageWidth']
        mask=[]
        for i,shape in enumerate(data['shapes']):
    
            label_name = shape['label']
            points = shape['points']
            x1,y1 = int(points[0][0]),int(points[0][1])
            x2,y2 = int(points[1][0]),int(points[1][1])
            print('coordinates are',x1,y1,x2,y2)
            mask.append(np.zeros((height,width)))
            mask[i][y1:y2,x1:x2] = 1
            print(np.sum(mask))
            result.append((label_name,np.sum(mask[i]*depth)/np.sum(mask[i])))
    return result 


if __name__ == '__main__':
    depth_file = './20200901201504909_image1.yuv_depth.npy'
    json_file = './20200901201504909_image1.yuv_rawL.json'
    result = get_depth_by_labelme(json_file,depth_file)
    print(result)