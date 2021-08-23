import os
import sys
sys.path.append("/home/lpy/paper/LiTS/dataset/")
from utils.common import *
sys.path.append("/home/lpy/paper/LiTS/model/Unet3D/")
from dataset.LitsMyself import Lits_DataSet
from Unet3D import Unet3D
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import metric 
import torchvision
from collections import OrderedDict

@torch.no_grad()
def test_model(model,device, dataload,thr):  
    batch_size = dataload.batch_size # 需要设置
    
    liver_dscs = []
    liver_accs = []
    liver_ppvs = []
    liver_sens = []
    liver_hausdorff_distances = []
    liver_mious = []
    liver_Accs = []
    
    tumor_dscs = []
    tumor_accs = []
    tumor_ppvs = []
    tumor_sens = []
    tumor_hausdorff_distances = []
    tumor_mious = []
    tumor_Accs = []

    for i,data in enumerate(dataload):
        input = data[0]
        target = data[1]
      
        inputs = input.float().to(device)
        target = target.float().to(device)
        outputs= model(inputs)
        if deepvision:
            outputs = torch.sigmoid(outputs[0])
        else:
            # outputs = torch.sigmoid(outputs[0]) # with edge
            outputs = torch.sigmoid(outputs)
        # ET evaluate
        for depth in range(outputs.shape[2]):
            # print(outputs.cpu()[:, 0, depth, :, :].detach(), target.cpu().detach()[:, 0, depth, :, :])
            liver_miou, liver_Acc = metric.get_miou(target.cpu().detach()[:, 0, depth, :, :],outputs.cpu()[:, 0, depth, :, :].detach(),thr)
            liver_dsc, liver_acc, liver_ppv, liver_sen, liver_hausdorff_distance = metric.m_metric(target.cpu().detach()[:, 0, depth, :, :], outputs.cpu().detach()[:, 0, depth, :, :], thr)
            liver_dscs.append(liver_dsc)
            liver_accs.append(liver_acc)
            liver_ppvs.append(liver_ppv)
            liver_sens.append(liver_sen)
            liver_hausdorff_distances.append(liver_hausdorff_distance)
            
            liver_mious.append(liver_miou)
            liver_Accs.append(liver_Acc)
        
        # liver evaluate
        for depth in range(outputs.shape[2]):
            tumor_miou, tumor_Acc = metric.get_miou(target.cpu().detach()[:, 1, depth, :, :],outputs.cpu()[:, 1, depth, :, :].detach(),thr)
            tumor_dsc, tumor_acc, tumor_ppv, tumor_sen, tumor_hausdorff_distance = metric.m_metric(target.cpu().detach()[:, 1, depth, :, :],outputs.cpu().detach()[:, 1, depth, :, :],thr)
            tumor_dscs.append(tumor_dsc)
            tumor_accs.append(tumor_acc)
            tumor_ppvs.append(tumor_ppv)
            tumor_sens.append(tumor_sen)
            tumor_hausdorff_distances.append(tumor_hausdorff_distance)
            tumor_mious.append(tumor_miou)
            tumor_Accs.append(tumor_Acc)
    # print(len(liver_mious),np.mean(liver_mious), np.nanmean(liver_mious), np.count_nonzero(liver_mious))
    return np.nanmean(liver_mious), np.nanmean(liver_Accs),np.nanmean(liver_dscs), np.nanmean(liver_accs), np.nanmean(liver_ppvs), np.nanmean(liver_sens), np.nanmean(liver_hausdorff_distances), \
        np.nanmean(tumor_mious), np.nanmean(tumor_Accs),np.nanmean(tumor_dscs), np.nanmean(tumor_accs), np.nanmean(tumor_ppvs), np.nanmean(tumor_sens), np.nanmean(tumor_hausdorff_distances)
       

def test(model,device,dataloader,thr):
    return test_model(model,device,dataloader,thr)

def load_model_checkpoints(model,checkpoint_path='./checkpoints/newSeUnet/latest.pth'):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

if __name__ == "__main__":
    batch_size = 1
    thr = 0.55
    deepvision = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_model_lossfunc = "LiTS_AMEA_res2block_fl"
    

    model = AMEA_res2block(1, 2).cuda()
   
    fixd_path  = '/home/data/LiTS/fixed_data'
    # dataset = Lits_DataSet([48, 256, 256],1,fixd_path,mode='val') 
    dataset = Lits_DataSet(16,1,fixd_path,mode='val') 
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,num_workers=1, shuffle=False)

    tmp_checkpoint_path = os.path.join("/home/lpy/paper/LiTS/checkpoints/fixedLabel2/", dataset_model_lossfunc+'/')
    # model = nn.DataParallel(model,device_ids=[0]).cuda()
    # 计算模型参数量
    param_count = sum(param.numel() for param in model.parameters())
    model.eval()
    params = sum(param.numel() for param in model.parameters()) / 1e6

    liver_DiceScore = 0
    liver_best_epoch = -1

    tumor_DiceScore = 0
    tumor_best_epoch = -1

    for tmp in range(41, 65):
        load_checkpoint_path = tmp_checkpoint_path + "epoch_{}.pth".format(tmp)
        print('#Params: %.1fM' % (params))
        print('*' * 15,'batch_sizes = {}'.format(batch_size),'*' * 15)
        load_model_checkpoints(model,load_checkpoint_path)
        liver_iou, liver_Accs, liver_dsc, liver_acc, liver_ppv, liver_sen, liver_hausdorff_distance,\
        tumor_iou, tumor_Accs, tumor_dsc, tumor_acc, tumor_ppv, tumor_sen, tumor_hausdorff_distance\
             = test(model,device,dataloader,thr)
        # liver_iou, liver_Accs, liver_dsc, liver_acc, liver_ppv, liver_sen, liver_hausdorff_distance = test(model,device,dataloader,thr)
       
             
        epoch_num = ((load_checkpoint_path.split('/'))[-1]).split('.')[0]
        if liver_dsc > liver_DiceScore:
            liver_DiceScore = liver_dsc
            liver_best_epoch = epoch_num

        if tumor_dsc > tumor_DiceScore:
            tumor_DiceScore = tumor_dsc
            tumor_best_epoch = epoch_num
        
        print('*' * 15,epoch_num + ':','*' * 15)
        print('*' * 15,"liver_mIoU:", liver_iou,'*' * 15)
        print('*' * 15,"liver_Accuracy:", liver_Accs,'*' * 15)
        print('*' * 15,"liver_DiceScore:", liver_dsc,'*' * 15)
        print('*' * 15,"liver_acc:", liver_acc,'*' * 15)
        print('*' * 15,"liver_ppv:", liver_ppv,'*' * 15)
        print('*' * 15,"liver_sen:", liver_sen,'*' * 15)
        print('*' * 15,"liver_hausdorff_distance:", liver_hausdorff_distance,'*' * 15)

        print('*' * 15,"tumor_mIoU:", tumor_iou,'*' * 15)
        print('*' * 15,"tumor_Accuracy:", tumor_Accs,'*' * 15)
        print('*' * 15,"tumor_DiceScore:", tumor_dsc,'*' * 15)
        print('*' * 15,"tumor_acc:", tumor_acc,'*' * 15)
        print('*' * 15,"tumor_ppv:", tumor_ppv,'*' * 15)
        print('*' * 15,"tumor_sen:", tumor_sen,'*' * 15)
        print('*' * 15,"tumor_hausdorff_distance:", tumor_hausdorff_distance,'*' * 15)

    print(liver_DiceScore, liver_best_epoch, tumor_DiceScore, tumor_best_epoch)