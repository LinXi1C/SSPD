# -*- coding:utf-8 -*-
# @File   : UBFC.py
# @Time   : 2022/11/26 23:10
# @Author : Zhang Xinyu
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image

class Dataset_UBFC_Offline(Dataset):
    def __init__(self, frame_path, person_name, length=70, maxOverlap=0, retain_prob=1):
        super().__init__()
        self.length = length
        self.maxOverlap = maxOverlap
        self.retain_prob = retain_prob
        self.person_name = person_name
        self.frame_list_root_path = frame_path
        self.frame_lists_total = os.listdir(self.frame_list_root_path)
        self.frame_lists = []
        self.prepare_data()

    def __getitem__(self, index):
        idx = index % len(self.frame_lists)
        frame_list_path = os.path.join(self.frame_list_root_path, self.frame_lists[idx])
        frame_list_origin = np.load(frame_list_path)

        # training sample generate...
        if self.maxOverlap == 0:
            overlap = 0
        else:
            overlap = np.random.randint(0, self.maxOverlap)
        if self.length > frame_list_origin.shape[0] - overlap - 1:
            start_clip1 = 0
            start_clip2 = 0
        else:
            start_clip1 = np.random.randint(0, frame_list_origin.shape[0] - self.length - overlap - 1)
            start_clip2 = start_clip1 + overlap
        end_clip1 = start_clip1 + self.length
        end_clip2 = start_clip2 + self.length
        frame_list_clip1 = frame_list_origin[start_clip1:end_clip1+1, :, :, :]
        frame_list_clip2 = frame_list_origin[start_clip2:end_clip2+1, :, :, :]
        frame_list_tf1 = np.zeros([frame_list_clip1.shape[0], 128, 128, 3], dtype=np.uint8)
        frame_list_tf2 = np.zeros([frame_list_clip2.shape[0], 128, 128, 3], dtype=np.uint8)

        ## online DA
        # random crop.
        frame_list_crop = self.My_RandomCrop(frame_list_clip1, crop_height=128, crop_width=128)
        is_hFlip = np.random.random() > 0.5
        is_vFlip = np.random.random() > 0.5
        transform = A.Compose([
            A.VerticalFlip(p=is_vFlip),
            A.HorizontalFlip(p=is_hFlip)
        ])
        for i in range(frame_list_crop.shape[0]):
            frame_list_tf1[i] = transform(image=frame_list_crop[i])['image']

        residual_list_tf1 = np.zeros([frame_list_tf1.shape[0]-1, frame_list_tf1.shape[1], frame_list_tf1.shape[2],
                                      frame_list_tf1.shape[3]], dtype=np.int16)
        for i in range(residual_list_tf1.shape[0]):
            residual_list_tf1[i, :, :, :] = frame_list_tf1[i+1, :, :, :].astype(np.int16) - \
                                            frame_list_tf1[i, :, :, :].astype(np.int16)

        # random mask.
        mask = np.random.binomial(n=1, p=self.retain_prob, size=(residual_list_tf1.shape[0], 128, 128, 1))
        residual_list_tf1 *= mask

        ## target DA
        is_hFlip = np.random.random() > 0.5
        is_vFlip = np.random.random() > 0.5
        transform = A.Compose([
            A.Resize(height=128, width=128, interpolation=Image.BICUBIC),
            A.VerticalFlip(p=is_vFlip),
            A.HorizontalFlip(p=is_hFlip)
        ])
        for i in range(frame_list_clip2.shape[0]):
            frame_list_tf2[i] = transform(image=frame_list_clip2[i])['image']

        residual_list_tf2 = np.zeros([frame_list_tf2.shape[0]-1, frame_list_tf2.shape[1], frame_list_tf2.shape[2],
                                      frame_list_tf2.shape[3]], dtype=np.int16)
        for i in range(residual_list_tf2.shape[0]):
            residual_list_tf2[i, :, :, :] = frame_list_tf2[i+1, :, :, :].astype(np.int16) - \
                                            frame_list_tf2[i, :, :, :].astype(np.int16)

        ## ndarray -> Tensor
        residual_list_DA1_return = torch.from_numpy(residual_list_tf1).float().permute((3, 0, 1, 2))
        residual_list_DA2_return = torch.from_numpy(residual_list_tf2).float().permute((3, 0, 1, 2))

        name = '_'.join([self.frame_lists[idx].split('_')[0], self.frame_lists[idx].split('_')[1], self.frame_lists[idx].split('_')[2],
                         'clip1', str(start_clip1), str(end_clip1), 'clip2', str(start_clip2), str(end_clip2)])
        return residual_list_DA1_return, residual_list_DA2_return, self.frame_lists[idx], start_clip1, end_clip1, name, overlap

    def __len__(self):
        return len(self.frame_lists)

    def prepare_data(self):
        for path in self.frame_lists_total:
            if path.split('_')[1] in self.person_name and int(path.split('_')[3]) - int(path.split('_')[2]) >= self.length:
                self.frame_lists.append(path)
        self.frame_lists.sort(key=lambda x: int(x.split('_')[0]))

    @staticmethod
    def My_RandomCrop(frame_list, crop_height, crop_width, h_start=random.random(), w_start=random.random()):
        frame_list_aug = np.zeros([frame_list.shape[0], 128, 128, 3], dtype=np.uint8)
        height, width = frame_list_aug.shape[1:3]
        y1 = int((height - crop_height + 1) * h_start)
        y2 = y1 + crop_height
        x1 = int((width - crop_width + 1) * w_start)
        x2 = x1 + crop_width
        for i in range(frame_list.shape[0]):
            frame_list_aug[i] = frame_list[i, y1:y2, x1:x2, :]
        return frame_list_aug



