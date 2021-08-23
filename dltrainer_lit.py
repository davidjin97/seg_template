import os
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from net_builder import build_net
import logging
from loss.loss import FocalLoss2d
from loss.loss import FocalLoss3d
from torch.utils.tensorboard import SummaryWriter
import cv2
import skimage.io
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from dataset.BrainSegDataset import BrainSegDataset
# from dataset.LiTS import Lits_DataSet 
from dataset.Lits import Lits_DataSet
# from dataset.LitsAug import Lits_DataSet
import torch.nn as nn 
from evaluate.metric import *
from utils.torchsummary import summary
logger = logging.getLogger('global')

class SegTrainer(object):
    def __init__(self, opt):
        super(SegTrainer, self).__init__()
        self.opt = opt
        self.initialize()

    def _prepare_dataset(self):
        if not self.opt.evaluate:
            # train_dataset = Lits_DataSet(self.opt.train_list,[48, 256, 256],1,self.opt.train_dir)
            # train_dataset = Lits_DataSet(16,1,self.opt.train_dir,mode='train')
            train_dataset = Lits_DataSet(self.opt.train_dir,self.opt.train_list,self.opt.depth)
            self.n_train_img = len(train_dataset)
            self.max_iter = self.n_train_img * self.opt.train_epoch // self.opt.batch_size // self.opt.world_size
            train_sampler = DistributedSampler(train_dataset)
            self.train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0, batch_size=self.opt.batch_size,
                                           pin_memory=True, sampler=train_sampler)
            logger.info('train with {} pair images'.format(self.n_train_img))

            self.val_loader = []
            self.opt.val_dir = self.opt.get('val_dir', '')
            if self.opt.val_dir != '':
                # val_dataset = Lits_DataSet(self.opt.test_list,[48, 256, 256],1,self.opt.test_dir)
                # val_dataset = Lits_DataSet(16,1,self.opt.test_dir,mode='val')
                val_dataset = Lits_DataSet(self.opt.test_dir,self.opt.test_list,self.opt.depth)
                self.n_val_img = len(val_dataset)
                val_sampler = DistributedSampler(val_dataset)
                self.val_loader = DataLoader(val_dataset, shuffle=False, num_workers=0, batch_size=1,
                                             pin_memory=True, sampler=val_sampler)
                logger.info('val with {} pair images'.format(self.n_val_img))

        # test_dataset = Lits_DataSet(self.opt.test_list,[48, 256, 256],1,self.opt.test_dir)
        # test_dataset = Lits_DataSet(16,1,self.opt.train_dir,mode='val')
        test_dataset = Lits_DataSet(self.opt.test_dir,self.opt.test_list,self.opt.depth)
        self.n_test_img = len(test_dataset)
        test_sampler = DistributedSampler(test_dataset)
        self.test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=1,
                                      pin_memory=True, sampler=test_sampler)
        logger.info('test with {} pair images'.format(self.n_test_img))
        logger.info('Build dataset done.')

    def _build_net(self):
        self.net = build_net(self.opt.net_name)(**self.opt.model).to(self.opt.rank)
        # 这边可以添加summary，格式化网络 -- todo

        if not self.opt.evaluate:
            if self.opt.bn == 'sync':
                try:
                    import apex
                    self.net = apex.parallel.convert_syncbn_model(self.net)
                except:
                    logger.info('not install apex. thus no sync bn')
            elif self.opt.bn == 'freeze':
                self.net = self.net.apply(freeze_bn)

        # 使用DDP       
        if not self.opt.tocaffe or not self.opt.toonnx:
            self.net = DDP(self.net,device_ids=[self.opt.rank])

        if self.opt.evaluate:
            self.load_state_keywise(self.opt.resume_model)
            logger.info('Load resume model from {}'.format(self.opt.resume_model))
        elif self.opt.pretrain_model == '':
            logger.info('Initial a new model...')
        else:
            if os.path.isfile(self.opt.pretrain_model):
                self.load_state_keywise(self.opt.pretrain_model)
                logger.info('Load pretrain model from {}'.format(self.opt.pretrain_model))
            else:
                logger.error('Can not find the specific model %s, initial a new model...', self.opt.pretrain_model)


        logger.info('Build model done.')

    def _build_optimizer(self):
        # construct optimizer
        if self.opt.optimizer.type == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), self.opt.lr_scheduler.base_lr,
                                             momentum=0.9, weight_decay=0.0005)
        elif self.opt.optimizer.type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), self.opt.lr_scheduler.base_lr)
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.lr_scheduler.base_lr, amsgrad=True)

        # construct lr_scheduler to adjust learning rate
        if self.opt.lr_scheduler.type == 'STEP':
            self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=self.opt.lr_scheduler.step_size,gamma=self.opt.lr_scheduler.gamma)


        # construct loss function
        if self.opt.loss.type == 'multi_scale':
            self.loss_function = MultiScaleLoss(**self.opt.loss.kwargs)
        elif self.opt.loss.type == 'focal2d':
            self.loss_function = FocalLoss2d(**self.opt.loss.kwargs)
        elif self.opt.loss.type == 'focal3d':
            self.loss_function = FocalLoss3d()
        elif self.opt.loss.type == 'kl':
            self.loss_function = nn.KLDivLoss()
        elif self.opt.loss.type == 'l1':
            self.loss_function = nn.L1Loss()
        elif self.opt.loss.type == 'ce':
            self.loss_function = nn.CrossEntropyLoss()
        # bce的shape需要一致
        elif self.opt.loss.type == 'bce':
            self.loss_function = nn.BCELoss()
        elif self.opt.loss.type == 'bcel':
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            logger.error('incorrect loss type {}'.format(self.opt.loss.type))

    def initialize(self):
        self._build_net()
        self._get_parameter_number()
        if not self.opt.tocaffe or not self.opt.toonnx:
            self._prepare_dataset()
            if not self.opt.evaluate:
                self._build_optimizer()


    def _get_parameter_number(self):
        total_num = sum(p.numel() for p in self.net.parameters())
        trainable_num = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logger.info(f'Total net parameters are {total_num//(1e6)}M, and weight file size is {trainable_num*4//(1024**2)}MB')


    def train(self):
        self.net.train()
        if self.opt.rank == 0:
            if not self.opt.log_directory:
                self.opt.log_directory = os.makedirs(self.opt.log_directory)
            writer = SummaryWriter(self.opt.log_directory)

        train_start_time = time.time()
        for epoch in range(self.opt.train_epoch):
            for iter_train, (image,mask) in enumerate(self.train_loader):
                # mask1 = F.interpolate(mask, size=(mask.shape[2] // 2, mask.shape[3] // 2, mask.shape[4] // 2))
                # mask2 = F.interpolate(mask, size=(mask.shape[2] // 4, mask.shape[3] // 4, mask.shape[4] // 4))
                # mask1 = mask1.cuda()
                # mask2 = mask2.cuda()

                image = image.cuda()
                mask = mask.cuda()
                output = self.net(image) 
                loss = self.loss_function(output,mask)
                # loss2 = self.loss_function(o1,mask1)
                # loss3 = self.loss_function(o2,mask2)
                # loss = loss1+loss2+loss3
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # logger.info(f'Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train}, the respective loss  is {loss1.item()}, {loss2.item()}, {loss3.item()}')
                logger.info(f'Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train}, the loss is {loss.item()}')
                if self.opt.rank == 0:
                    writer.add_scalar(f'Loss/train',loss.item(),iter_train+epoch*len(self.train_loader))
            self.lr_scheduler.step()

            if self.opt.rank == 0 and ((epoch+1)%self.opt.save_every_epoch)==0:
                self.save_checkpoint({'epoch': self.opt.train_epoch,
                                    'arch': self.opt.net_name,
                                    'state_dict': self.net.state_dict(),
                                    }, f'epoch_{epoch+1}_model_final.pth')
                logger.info(f'Start evalute atEpoch: {epoch+1}/{self.opt.train_epoch}')
                self.test()
                self.net.train()
        logger.info('Finish training, cost time: {} h'.format((time.time()-train_start_time)/3600))

    def test(self):
        self.net.eval()
        thr = 0.55
        liver_iou, liver_Accs, liver_dsc, liver_acc, liver_ppv, liver_sen, liver_hausdorff_distance,\
        tumor_iou, tumor_Accs, tumor_dsc, tumor_acc, tumor_ppv, tumor_sen, tumor_hausdorff_distance\
             = test(self.net,self.test_loader,thr)
        logger.info(f"{'*'*15} liver_mIoU: {liver_iou},{'*'*15}")
        logger.info(f"{'*'*15} liver_Accuracy: {liver_Accuracy},{'*'*15}")
        logger.info(f"{'*'*15} liver_DiceScore: {liver_DiceScore},{'*'*15}")
        logger.info(f"{'*'*15} liver_acc: {liver_acc},{'*'*15}")
        logger.info(f"{'*'*15} liver_ppv: {liver_ppv},{'*'*15}")
        logger.info(f"{'*'*15} liver_sen: {liver_sen},{'*'*15}")
        logger.info(f"{'*'*15} liver_hausdorff_distance: {liver_hausdorff_distance},{'*'*15}")

        logger.info(f"{'*'*15} tumor_mIoU: {liver_iou},{'*'*15}")
        logger.info(f"{'*'*15} tumor_Accuracy: {liver_Accuracy},{'*'*15}")
        logger.info(f"{'*'*15} tumor_DiceScore: {liver_DiceScore},{'*'*15}")
        logger.info(f"{'*'*15} tumor_acc: {liver_acc},{'*'*15}")
        logger.info(f"{'*'*15} tumor_ppv: {liver_ppv},{'*'*15}")
        logger.info(f"{'*'*15} tumor_sen: {liver_sen},{'*'*15}")
        logger.info(f"{'*'*15} tumor_hausdorff_distance: {liver_hausdorff_distance},{'*'*15}")

        pass

    def val(self):
        self.net.eval()
        pass

    def save_checkpoint(self, state_dict, filename='checkpoint.pth'):
        torch.save(state_dict, os.path.join(self.opt.train_output_directory, filename))

    def convert2caffe(self, save_name):
        test_height = self.opt.get('test_height', 576)
        test_width = self.opt.get('test_width', 960)
        from spring.nart.tools.pytorch import convert_mode, convert
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        self.net.eval()
        with convert_mode():
            convert.convert(self.net, [(6,test_height,test_width)], save_name, verbose=True)

    def convert2onnx(self, save_name):
        test_height = self.opt.get('test_height', 576)
        test_width = self.opt.get('test_width', 960)
        from spring.nart.tools.pytorch import convert_mode, convert
        import torch.onnx as torch_onnx
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        self.net.eval()
        with convert_mode():
            # convert.convert(self.net, [(6,test_height,test_width)], save_name, verbose=True)
            torch_onnx.export(self.net,torch.randn(1,6,test_height,test_width).cuda(),save_name+'.onnx',verbose=True)

    def load_state_keywise(self, model_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.opt.rank}
        resume_dict = torch.load(model_path, map_location=map_location)
        if 'state_dict' in resume_dict.keys():
            resume_dict = resume_dict['state_dict']
        self.net.load_state_dict(resume_dict)
