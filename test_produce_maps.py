import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import argparse


import cv2
from Code.lib.model import Net
from Code.utils.data import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test


# load the model
model = Net(32, 50)
model.cuda()

model.load_state_dict(torch.load(''))#pth file path
model.eval()

# test
test_datasets = ['ReDWeb-S', 'NJUD', 'NLPR', 'DUT-RGBD', 'STERE1000']

test_datasets = ['ReDWeb-S', 'NJUD', 'NLPR', 'DUT-RGBD', 'STERE1000']

for dataset in test_datasets:
    save_path = './testMaps' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/test_data/test_images/'
    gt_root = dataset_path + dataset + '/test_data/test_masks/'
    depth_root = dataset_path + dataset + '/test_data/test_depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    img_num = len(test_loader)
    time_s = time.time()
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        pre_res = model(image, depth)
        res = pre_res[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, res * 255)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))
    print('Test Done!')