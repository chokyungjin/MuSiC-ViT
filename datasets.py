import os
import sys
import numpy as np
import torch
import random
import glob

from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

from config import parse_arguments
from PIL import Image

import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

from skimage import exposure


class ClassPairDataset(Dataset):
    def __init__(self, input_path, dataset, mode, margin=False, sample_data=None, fov=False, mean=0.2, std=0.4, aug=False,transform=None):
        if dataset == 'toy':
            input_path = input_path + '_toy'
        
        self.input_path = os.path.join(input_path, '{}set/512'.format(mode))
        self.disease_label_path = os.path.join(input_path, 'label_csv/disease.json')
        self.aug = aug
        self.fov = fov
        self.mean = mean
        self.std = std
        self.sample_data = sample_data
        self.margin = margin
        with open(self.disease_label_path, "r") as f:
            self.disease_label = json.load(open(self.disease_label_path))
        
        if self.margin is not None:
            print("[*] Margin true")
        else:
            print("[*] Margin false")
        if self.fov is not None:
            print("FOV true")
            if self.sample_data is not None:
                print("[*] Sample data loaded")
                json_name = './json/4class_datasets_{}_sample_512_fov_{}.json'.format(dataset,mode)
                #json_name = './json/4class_datasets_{}_512_fov_{}_Supcon_sampling.json'.format(dataset,mode)
                # json_name = './json/4class_datasets_{}_512_fov_{}_Supcon_sampling_change_predict.json'.format(dataset,mode)
            else:
                json_name = './json/4class_datasets_{}_512_fov_{}.json'.format(dataset,mode)
                # json_name = './json/internval_validation_add_fov.json'  # ER
        else:
            json_name = './json/4class_datasets_{}_512_{}.json'.format(dataset,mode)
        if os.path.exists(json_name) is True:
            print('[*] {} is already exist. Loading Json from {}'.format(json_name, json_name))
            with open(json_name, "r") as f:
                self.samples = json.load(f)
        else:
            print('[*] There is no {}. Start making new Json'.format(json_name, json_name))
            self.samples = self._make_dataset(mode)
            with open(json_name, "w") as f:
                json.dump(self.samples, f)
        
        if mode == 'train':
            if self.aug is not None:
                print("[*] Augmentation On")
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.MotionBlur(p=0.2),
                        A.Sharpen(p=0.2),
                        ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
                    A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        ], p=0.2),
                    A.OneOf([
                        A.CLAHE(clip_limit=4.0),
                        A.Equalize(),
                        ], p=0.2),
                    A.OneOf([
                        A.GaussNoise(p=0.2),
                        A.MultiplicativeNoise(p=0.2),
                        ], p=0.2),
                    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3),
                    A.Normalize(mean=(self.mean), std=(self.std,)),
                    ToTensorV2(),
                    ], additional_targets={'image0': 'image' , 'image1': 'image' , 'image2': 'image'})

            else:
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=(self.mean), std=(self.std,)),
                    ToTensorV2(),
                    ], additional_targets={'image0': 'image' , 'image1': 'image' , 'image2': 'image'})

        else:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=(self.mean,), std=(self.std,)),
                ToTensorV2(),
                ], additional_targets={'image0': 'image' , 'image1': 'image' , 'image2': 'image'})
                
    def _find_disease_label(self, exam_id):
        if exam_id in self.disease_label['normal']:
            return 0 #normal
        elif exam_id in self.disease_label['abnormal']:
            return 1 #abnormal
        else:
            return 2

    def _check_crop_label(self, crop_label):
        x_start, x_end, y_start, y_end = crop_label
        x_margin = int(511-x_start)
        y_margin = int(y_end-y_start)
        if x_margin < 300:
            x_start = 47
        if y_margin < 300:
            y_start, y_end = 55, 453
        return [x_start, x_end, y_start, y_end]

    def _check_crop_label_margin(self, crop_label, margin=True, ratio=0.08):
        x_start, x_end, y_start, y_end = crop_label
        x_margin = int(511-x_start)
        y_margin = int(y_end-y_start)

        if x_margin < 300:
            x_start, x_end = 47, 445
        if y_margin < 300:
            y_start, y_end = 55, 453

        if margin is True:
            margin = int((x_end - x_start) * ratio // 2)
            x_start -= margin
            x_end += margin

            margin = int((y_end - y_start) * ratio // 2)
            y_start -= margin
            y_end += margin


        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0
        if x_end > 511:
            x_end = 511     
        if y_end > 511:
            y_end = 511  

        return [x_start, x_end, y_start, y_end]

    def __getitem__(self, idx):
        if self.fov is not None:
            if self.margin is not None:
                x_min, _, y_min, y_max = self._check_crop_label_margin(self.samples['fov'][idx][0])
            else:
                x_min, _, y_min, y_max = self._check_crop_label(self.samples['fov'][idx][0])
            base_img = np.array(Image.open(self.samples['imgs'][idx][0])) 
            base_img = base_img[x_min:, y_min:y_max]
            
            if self.margin is not None:
                x_min, _, y_min, y_max = self._check_crop_label_margin(self.samples['fov'][idx][1])
            else:
                x_min, _, y_min, y_max = self._check_crop_label(self.samples['fov'][idx][1])
            pair_img = np.array(Image.open(self.samples['imgs'][idx][1]))
            pair_img = pair_img[x_min:, y_min:y_max]
        else:
            base_img = np.array(Image.open(self.samples['imgs'][idx][0]))
            pair_img = np.array(Image.open(self.samples['imgs'][idx][1]))
            
        transformed = self.transform(image=base_img, image0=pair_img)
        
        base_img = transformed['image']
        base_img = self._catch_exception(base_img)

        pair_img = transformed['image0']
        pair_img = self._catch_exception(pair_img)
        
        change_labels = self.samples['change_labels'][idx]
        disease_labels = self.samples['disease_labels'][idx]
        patient_name = self.samples['imgs'][idx][0]#.split('/')[-2]
        
        return base_img, pair_img, change_labels, disease_labels, patient_name
            
    def __len__(self):
        return len(self.samples['change_labels'])
    
    def _catch_exception(self, img):
        return img[0, :, :].unsqueeze(0) if img.shape[0] == 3 else img

    def _get_change_label_num(self, label, label_list):
        specific_labels = []
        for i in label_list:
            if label == i:
                specific_labels.append(i)
        return len(specific_labels)
                
    def _get_disease_label_num(self, label, label_list):
        specific_labels = []
        for i in label_list:
            for j in i:
                if label == j:
                    specific_labels.append(i)
        return len(specific_labels)

    def get_data_property(self):
        if len(self.samples['change_labels']):
            print('images(pair): {}\nlabels(change): {}\nlabels(nochange): {}\nlabels(normal): {}\nlabels(abnormal): {}\nlabels(unknown): {}'.format(
                    len(self.samples['imgs']), 
                    self._get_change_label_num(0, self.samples['change_labels']),
                    self._get_change_label_num(1, self.samples['change_labels']),
                    self._get_disease_label_num(0, self.samples['disease_labels']),
                    self._get_disease_label_num(1, self.samples['disease_labels']),
                    self._get_disease_label_num(2, self.samples['disease_labels']),
                    )
                    )
                    
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])