import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.data_util import max_normalize
from basicsr.data.transforms import augment, sar_augment, add_noise, view_sar
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from torch.utils import data as data

from basicsr.data.transforms import paired_random_crop


class SarDataset(data.Dataset):
    def __init__(self, opt):
        super(SarDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']  # /home/fyt/SARdata

        self.lq_paths = []
        self.gt_paths = []
        with open(self.opt['meta_info'], 'r', encoding='utf-8') as fin:
            for line in fin:
                lq_path, gt_path = line.strip().split(',')  # ori/1_20160213.rmli.npy
                self.lq_paths.append(os.path.join(self.gt_folder, lq_path))
                self.gt_paths.append(os.path.join(self.gt_folder, gt_path))

        self.crop_pad_size = opt['crop_pad_size']
        self.is_view_to_gray = opt.get('is_view_to_gray', False)

    def __getitem__(self, index):
        # if self.file_client is None:
        #     self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]

        img_ori_gt = np.load(gt_path)
        img_ori_lq = np.load(lq_path)  # H,W
        
        if not self.opt.get('is_val', False):
            img_ori_lq, img_ori_gt = sar_augment(img_ori_lq, img_ori_gt, 
                                                 speckle=self.opt.get('use_speckle', False), 
                                                 contrast=self.opt.get('use_contrast', False),
                                                 brightness=self.opt.get('use_brightness', False), 
                                                 gamma=self.opt.get('use_gamma', False), 
                                                 elastic=self.opt.get('use_elastic', False))
        
        img_gt = np.sqrt(img_ori_gt)[:, :, np.newaxis]
        img_lq = np.sqrt(img_ori_lq)[:, :, np.newaxis]
        
        if self.is_view_to_gray:
            img_gt, _, _ = max_normalize(img_gt)  # amp to norm
            img_lq, _, _ = max_normalize(img_lq)
            img_gt = view_sar(img_gt)  # norm to gray
            img_lq = view_sar(img_lq)
        
        img_lq, lq_min, lq_max = max_normalize(img_lq)
        gt_min, gt_max = lq_min, lq_max
        
        img_gt[np.isnan(img_gt)] = 0
        img_gt = np.abs((img_gt - lq_min) / (lq_max - lq_min))

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.crop_pad_size, 1.0, gt_path=gt_path)

        # HWC to CHW, numpy to tensor
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        return_d = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path,
                    # 'lq_ori': img_ori_lq, 'gt_ori': img_ori_gt,
                    'lq_min': lq_min, 'lq_max': lq_max, 'gt_min': gt_min, 'gt_max': gt_max}
        return return_d

    def __len__(self):
        return len(self.gt_paths)


class SarEvalSdOnNrdDataset(SarDataset):
    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]

        img_ori_gt = np.load(gt_path)
        img_ori_lq = np.load(lq_path)  # H,W
        
        def _view(img):
            _sum = np.sum(img)
            _len = len(np.nonzero(img)[0])
            scale = 0.3 / (_sum / _len)
            img = img * scale
            img = np.where(img > 1, 1, img)
            # img = (img * 255.).astype(np.uint8)
            return img

        img_gt = np.sqrt(img_ori_gt)[:, :, np.newaxis]
        img_lq = np.sqrt(img_ori_lq)[:, :, np.newaxis]
        
        # point
        img_gt = _view(img_gt)
        img_lq = _view(img_lq)
        
        img_lq, lq_min, lq_max = max_normalize(img_lq)
        gt_min, gt_max = lq_min, lq_max
        
        img_gt[np.isnan(img_gt)] = 0
        img_gt = np.abs((img_gt - lq_min) / (lq_max - lq_min))

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.crop_pad_size, 1.0, gt_path=gt_path)

        # HWC to CHW, numpy to tensor
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        return_d = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path,
                    # 'lq_ori': img_ori_lq, 'gt_ori': img_ori_gt,
                    'lq_min': lq_min, 'lq_max': lq_max, 'gt_min': gt_min, 'gt_max': gt_max, 
                    'is_synthetic': True}
        return return_d

    def __len__(self):
        return len(self.gt_paths)


class SarSyntheticDataset(data.Dataset):
    def __init__(self, opt):
        super(SarSyntheticDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']  # /home/fyt/SARdata

        self.lq_paths = []
        self.gt_paths = []
        with open(self.opt['meta_info'], 'r', encoding='utf-8') as fin:
            for line in fin:
                lq_path, gt_path = line.strip().split(',')
                self.lq_paths.append(os.path.join(self.gt_folder, lq_path))
                self.gt_paths.append(os.path.join(self.gt_folder, gt_path))

        self.crop_pad_size = opt['crop_pad_size']

    def _get_img_pair(self, index):
        sd_lq_path = self.lq_paths[index]
        sd_gt_path = self.gt_paths[index]

        try:
            sd_ori_lq = cv2.imread(sd_lq_path)
            sd_ori_lq = cv2.cvtColor(sd_ori_lq, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]  # L=1 is too speckled, HW

            sd_ori_gt = cv2.imread(sd_gt_path)
            sd_ori_gt = cv2.cvtColor(sd_ori_gt, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        except:
            print('error reading image: ', sd_lq_path)
            return self._get_img_pair((index + 1) % len(self.lq_paths))
        return sd_lq_path, sd_gt_path, sd_ori_lq, sd_ori_gt

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        sd_lq_path, sd_gt_path, sd_ori_lq, sd_ori_gt = self._get_img_pair(index)

        norm_size = self.crop_pad_size
        if sd_ori_gt.shape[0] < norm_size or sd_ori_gt.shape[1] < norm_size:
            sd_ori_gt = cv2.resize(sd_ori_gt, (norm_size, norm_size))[:, :, np.newaxis]
            sd_ori_lq = cv2.resize(sd_ori_lq, (norm_size, norm_size))[:, :, np.newaxis]

        sd_lq, sd_lq_min, sd_lq_max = max_normalize(sd_ori_lq)
        sd_lq = sd_lq.astype(np.float32)
        sd_gt_min, sd_gt_max = sd_lq_min, sd_lq_max
        sd_gt = np.abs((sd_ori_gt - sd_lq_min) / (sd_lq_max - sd_lq_min)).astype(np.float32)

        sd_gt, sd_lq = paired_random_crop(sd_gt, sd_lq, self.crop_pad_size, 1, gt_path=sd_gt_path)

        (sd_gt, sd_lq) = (augment([sd_gt, sd_lq], self.opt['use_hflip'], self.opt['use_rot']))

        # HWC to CHW, numpy to tensor
        (sd_gt, sd_lq) = map(lambda x: img2tensor([x], bgr2rgb=True, float32=True)[0],
                                                   (sd_gt, sd_lq)
                                                   )

        return_d = {'lq': sd_lq, 'gt': sd_gt, 'lq_path': sd_gt_path,
                    # 'lq_ori': sd_ori_lq.squeeze(-1), 'gt_ori': sd_ori_gt.squeeze(-1),  # HW
                    'lq_min': sd_lq_min, 'lq_max': sd_lq_max, 'gt_min': sd_gt_min, 'gt_max': sd_gt_max, 
                    'is_synthetic': True}
        return return_d

    def __len__(self):
        return len(self.gt_paths)


class SarJointDataset(data.Dataset):
    def __init__(self, opt):
        super(SarJointDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.nrd_folder = opt['dataroot_gt']  # near real dataset, /home/fyt/SARdata
        self.sd_folder = opt['synthetic_dataroot']  # synthetic dataset
        sd_ext = opt.get('synthetic_ext', 'tif')  # synthetic dataset

        self.nrd_lq_paths = []
        self.nrd_gt_paths = []
        with open(self.opt['meta_info'], 'r', encoding='utf-8') as fin:
            for line in fin:
                lq_path, gt_path = line.strip().split(',')  # ori/1_20160213.rmli.npy
                self.nrd_lq_paths.append(os.path.join(self.nrd_folder, lq_path))
                self.nrd_gt_paths.append(os.path.join(self.nrd_folder, gt_path))

        self.sd_gt_paths = glob.glob(self.sd_folder + f'/**/*.{sd_ext}', recursive=True)
        self.sd_gt_len = len(self.sd_gt_paths)

        self.crop_pad_size = opt['crop_pad_size']
        self.train_crop_size = opt.get('train_crop_size', 32)
        self.is_view_to_gray = opt.get('is_view_to_gray', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # image range: [0, 1], float32. intensity
        nrd_gt_path = self.nrd_gt_paths[index]
        nrd_lq_path = self.nrd_lq_paths[index]

        while True:
            nrd_ori_gt = np.load(nrd_gt_path)
            nrd_ori_lq = np.load(nrd_lq_path)  # H,W
            if len(nrd_ori_lq.shape) == 2:
                nrd_ori_lq = nrd_ori_lq[:, :, np.newaxis]
                nrd_ori_gt = nrd_ori_gt[:, :, np.newaxis]
            nrd_ori_gt, nrd_ori_lq = paired_random_crop(nrd_ori_gt, nrd_ori_lq, self.crop_pad_size, 1, gt_path=nrd_gt_path)

            # image range: [0, 255], float32. amplitude
            sd_gt_path = self.sd_gt_paths[random.randint(0, self.sd_gt_len - 1)]

            sd_ori_gt = cv2.imread(sd_gt_path)
            if sd_ori_gt is None or len(sd_ori_gt) == 0:
                index = (index+1) % len(self.nrd_gt_paths)
                continue
            sd_ori_gt = cv2.cvtColor(sd_ori_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sd_ori_lq = add_noise(sd_ori_gt).astype(np.float32)  # L=1 is too speckled
            if len(sd_ori_lq.shape) == 2:
                sd_ori_lq = sd_ori_lq[:, :, np.newaxis]
                sd_ori_gt = sd_ori_gt[:, :, np.newaxis]
            sd_ori_gt, sd_ori_lq = paired_random_crop(sd_ori_gt, sd_ori_lq, self.crop_pad_size, 1, gt_path=nrd_gt_path)
            # if len(sd_ori_lq) == 0:
            #     index = (index+1) % len(self.nrd_gt_paths)
            #     continue
            try:
                sd_lq, sd_lq_min, sd_lq_max = max_normalize(sd_ori_lq)
            except ValueError:  # ValueError: zero-size array to reduction operation minimum which has no identity
                index = (index+1) % len(self.nrd_gt_paths)
                continue
            break
        
        # intensity -> amplitude
        nrd_gt = np.sqrt(nrd_ori_gt)
        nrd_lq = np.sqrt(nrd_ori_lq)
        
        if self.is_view_to_gray:
            nrd_gt, _, _ = max_normalize(nrd_gt)  # amp to norm
            nrd_lq, _, _ = max_normalize(nrd_lq)
            nrd_gt = view_sar(nrd_gt)  # norm to gray
            nrd_lq = view_sar(nrd_lq)
        
        nrd_lq, nrd_lq_min, nrd_lq_max = max_normalize(nrd_lq)  # [0,1]
        nrd_gt_min, nrd_gt_max = nrd_lq_min, nrd_lq_max
        
        nrd_gt[np.isnan(nrd_gt)] = 0
        nrd_gt = np.abs((nrd_gt - nrd_lq_min) / (nrd_lq_max - nrd_lq_min))

        sd_lq, sd_lq_min, sd_lq_max = max_normalize(sd_ori_lq)
        sd_gt_min, sd_gt_max = sd_lq_min, sd_lq_max
        sd_gt = np.abs((sd_ori_gt - sd_lq_min) / (sd_lq_max - sd_lq_min))

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        nrd_gt_i, nrd_lq_i = paired_random_crop(nrd_gt, nrd_lq, self.train_crop_size, 1, gt_path=nrd_gt_path)
        nrd_gt_j, nrd_lq_j = paired_random_crop(nrd_gt, nrd_lq, self.train_crop_size, 1, gt_path=nrd_gt_path)

        sd_gt_i, sd_lq_i = paired_random_crop(sd_gt, sd_lq, self.train_crop_size, 1, gt_path=sd_gt_path)
        sd_gt_j, sd_lq_j = paired_random_crop(sd_gt, sd_lq, self.train_crop_size, 1, gt_path=sd_gt_path)

        (nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
         sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j) = (augment(
            [nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
             sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j],
            self.opt['use_hflip'], self.opt['use_rot'])
        )

        # HWC to CHW, numpy to tensor
        (nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
         sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j) = map(lambda x: img2tensor([x], bgr2rgb=True, float32=True)[0],
                                                   (nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
                                                    sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j)
                                                   )

        return_d = {'nrd_lq_i': nrd_lq_i, 'nrd_gt_i': nrd_gt_i, 'nrd_lq_j': nrd_lq_j, 'nrd_gt_j': nrd_gt_j,
                    'sd_lq_i': sd_lq_i, 'sd_gt_i': sd_gt_i, 'sd_lq_j': sd_lq_j, 'sd_gt_j': sd_gt_j,

                    'nrd_lq_path': nrd_lq_path, 'sd_gt_path': sd_gt_path,

                    'nrd_lq_min': nrd_lq_min, 'nrd_lq_max': nrd_lq_max,
                    'nrd_gt_min': nrd_gt_min, 'nrd_gt_max': nrd_gt_max,

                    'sd_lq_min': sd_lq_min, 'sd_lq_max': sd_lq_max,
                    'sd_gt_min': sd_gt_min, 'sd_gt_max': sd_gt_max,
                    }
        return return_d

    def __len__(self):
        return len(self.nrd_gt_paths)


class SarJointSyntheticDataset(SarSyntheticDataset):
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # image range: [0, 1], float32. intensity

        sd_lq_path, sd_gt_path, sd_ori_lq, sd_ori_gt = self._get_img_pair(index)

        norm_size = self.crop_pad_size
        if sd_ori_gt.shape[0] < norm_size or sd_ori_gt.shape[1] < norm_size:
            sd_ori_gt = cv2.resize(sd_ori_gt, (norm_size, norm_size))[:, :, np.newaxis]
            sd_ori_lq = cv2.resize(sd_ori_lq, (norm_size, norm_size))[:, :, np.newaxis]

        sd_lq, sd_lq_min, sd_lq_max = max_normalize(sd_ori_lq)
        sd_gt_min, sd_gt_max = sd_lq_min, sd_lq_max
        sd_gt = np.abs((sd_ori_gt - sd_lq_min) / (sd_lq_max - sd_lq_min))

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        nrd_gt_i, nrd_lq_i = paired_random_crop(sd_gt, sd_lq, self.crop_pad_size, 1, gt_path=sd_gt_path)
        nrd_gt_j, nrd_lq_j = paired_random_crop(sd_gt, sd_lq, self.crop_pad_size, 1, gt_path=sd_gt_path)

        sd_gt_i, sd_lq_i = paired_random_crop(sd_gt, sd_lq, self.crop_pad_size, 1, gt_path=sd_gt_path)
        sd_gt_j, sd_lq_j = paired_random_crop(sd_gt, sd_lq, self.crop_pad_size, 1, gt_path=sd_gt_path)

        (nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
         sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j) = (augment(
            [nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
             sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j],
            self.opt['use_hflip'], self.opt['use_rot'])
        )

        # HWC to CHW, numpy to tensor
        (nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
         sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j) = map(lambda x: img2tensor([x], bgr2rgb=True, float32=True)[0],
                                                   (nrd_gt_i, nrd_lq_i, nrd_gt_j, nrd_lq_j,
                                                    sd_gt_i, sd_lq_i, sd_gt_j, sd_lq_j)
                                                   )

        return_d = {'nrd_lq_i': nrd_lq_i, 'nrd_gt_i': nrd_gt_i, 'nrd_lq_j': nrd_lq_j, 'nrd_gt_j': nrd_gt_j,
                    'sd_lq_i': sd_lq_i, 'sd_gt_i': sd_gt_i, 'sd_lq_j': sd_lq_j, 'sd_gt_j': sd_gt_j,

                    'nrd_lq_path': sd_gt_path, 'sd_gt_path': sd_gt_path,

                    'nrd_lq_min': sd_lq_min, 'nrd_lq_max': sd_lq_max,
                    'nrd_gt_min': sd_gt_min, 'nrd_gt_max': sd_gt_max,

                    'sd_lq_min': sd_lq_min, 'sd_lq_max': sd_lq_max,
                    'sd_gt_min': sd_gt_min, 'sd_gt_max': sd_gt_max,
                    'is_synthetic': True
                    }
        return return_d


if __name__ == '__main__':
    _dataset = SarJointDataset({
        'dataroot_gt': 'datasets/SARdata',
        'synthetic_dataroot': 'datasets/AID/Images',
        'synthetic_ext': 'jpg',
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'meta_info': 'datasets/train.txt',
        'crop_pad_size': 64,
        'use_hflip': True,
        'use_shuffle': True,
        'use_hflip': True,
        'use_rot': True,
    })
    for i in range(len(_dataset)):
        if i > 2:
            break
        print(i)
        print(_dataset.__getitem__(i))
        