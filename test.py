import sys
import os
import random
import numpy as np
from config import parse_arguments
from datasets_with_UNet import ClassPairDataset

from models.siamese_CMT_with_UNet_ACM import siamese_CMT_with_UNet_ACM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import time
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import json
import itertools
from utils import *
from gradcam import *

def test(args, data_loader, model, grad_cam, device, log_dir):

    print('[*] Test Phase')
    model.eval()
    model.requires_grad_(False)
    correct = 0
    total = 0
        
    overall_output = []
    overall_pred = []
    overall_gt = []
    overall_pat = []
    iter_ = 0

    idx = 0

    for base, fu, base_2, fu_2, labels, patient_name in tqdm(iter(data_loader)):
        
        base = base.to(device)
        fu = fu.to(device)
        
        if args.hist_equal == 'True':
            base_2 = base_2.to(device)
            fu_2 = fu_2.to(device)
        else:
            base_2 = base
            fu_2 = fu
            
        labels = labels.to(device)
        
        '''
        Unet encoder cam map add
        '''
        base_cam_result = torch.zeros_like(base)
        fu_cam_result = torch.zeros_like(fu)
        B = base.shape[0]
                
        for i in range(B):
            grayscale_cam_base, _ = grad_cam(base_2[i:i+1,...])
            if torch.isnan(grayscale_cam_base).any():
                grayscale_cam_base = torch.zeros_like(grayscale_cam_base).to(device)
            base_cam_result[i,...] = (grayscale_cam_base)

            grayscale_cam_fu, _ = grad_cam(fu_2[i:i+1,...])
            if torch.isnan(grayscale_cam_fu).any():
                grayscale_cam_fu = torch.zeros_like(grayscale_cam_fu).to(device)
            fu_cam_result[i,...] = (grayscale_cam_fu)
            
        with torch.no_grad():
            _, _, outputs, _ = model(base, fu, base_cam_result, fu_cam_result)
            outputs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            new_labels = []
            for i in range(labels.shape[0]):
                if np.isnan(outputs[i, 1].cpu().detach().item()).any():
                    out = 0
                else:
                    out = outputs[i, 1].cpu().detach().item()
                    
                new_labels.append(out)

            preds_cpu = preds.cpu().detach().numpy().tolist()
            labels_cpu = labels.cpu().detach().numpy().tolist()
            overall_output += new_labels
            overall_pred += preds_cpu
            overall_gt += labels_cpu
            overall_pat += patient_name

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        
        iter_ += 1

        idx += base.shape[0]
    
    print('[*] Test Acc: {:5f}'.format(100.*correct/total))
    
    tn, fp, fn, tp = confusion_matrix(overall_gt, overall_pred).ravel()
    
    save_confusion_matrix(confusion_matrix(overall_gt, overall_pred), ['Change','No-Change'], log_dir)
    save_results_metric(tn, tp, fn, fp, correct, total, log_dir)
    save_roc_auc_curve(overall_gt, overall_output, log_dir)
    save_csv(overall_pat, overall_gt, overall_pred, overall_output, log_dir)
        

def main(args):
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        
    # 0. device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}_{}'.format(today, args.message, args.dataset)
    
    log_dir = os.path.join(args.log_dir, folder_name)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()

    # make datasets & dataloader (train & test)
    print('[*] prepare datasets & dataloader...')

    if args.fov == 'True':
        test_datasets = ClassPairDataset(args.test_path, dataset=args.dataset, fov=args.fov, 
                                        sample_data=args.sample_data, 
                                        margin=args.margin, mode='test')
    else:
        test_datasets = ClassPairDataset(args.test_path, dataset=args.dataset, 
                            fov=None, sample_data=None, aug=None, margin=args.margin, mode='test')
        
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=2, 
                            num_workers=4, pin_memory=False, shuffle=False)
     
    # select network
    
    print('[*] build network...')
    unet_model = torch.load('./radiography_model/best_model.pth')

    # select network
    if args.backbone =='CMT_Ti':
        model = siamese_CMT_with_UNet_ACM(in_channels = 2,
                            stem_channels = 16,
                            cmt_channelses = [46, 92, 184, 368],
                            pa_channelses = [46, 92, 184, 368],
                            R = 3.6,
                            repeats = [2, 2, 10, 2],
                            input_size = 512,
                            sizes = [128, 64, 32, 16],
                            patch_ker=2,
                            patch_str=2,
                            num_classes = 2)
        
    elif args.backbone == 'CMT_XS':
        model = siamese_CMT_with_UNet_ACM(in_channels = 2,
                            stem_channels = 16,
                            cmt_channelses = [52, 104, 208, 416],
                            pa_channelses = [52, 104, 208, 416],
                            R = 3.8,
                            repeats = [3, 3, 12, 3],
                            input_size = 512,
                            patch_ker=2,
                            patch_str=2,
                            num_classes = 2)
        
    elif args.backbone == 'CMT_S':
        model = siamese_CMT_with_UNet_ACM(in_channels = 2,
                            stem_channels = 32,
                            cmt_channelses = [64, 128, 256, 512],
                            pa_channelses = [64, 128, 256, 512],
                            R = 4.,
                            repeats = [3, 3, 16, 3],
                            input_size = 512,
                            patch_ker=2,
                            patch_str=2,
                            num_classes = 2)
    
    elif args.backbone == 'CMT_B':
        model = siamese_CMT_with_UNet_ACM(in_channels = 2,
                            stem_channels = 38,
                            cmt_channelses = [76, 152, 304, 608],
                            pa_channelses = [76, 152, 304, 608],
                            R = 4.,
                            repeats = [4, 4, 20, 4],
                            input_size = 512,
                            sizes = [128, 64, 32, 16],
                            patch_ker=2,
                            patch_str=2,
                            num_classes = 2)

    print("[*] model")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        unet_model = nn.DataParallel(unet_model)

    model = model.to(device)
    unet_model = unet_model.to(device)
    

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.AdamW(model.module.parameters(), lr = args.lr,
            weight_decay = args.weight_decay)
    
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr,
            weight_decay = args.weight_decay)

    
    if torch.cuda.device_count() > 1:
        target_layer = unet_model.module.encoder.features.denseblock4.denselayer16.conv2
    else:
        target_layer = unet_model.encoder.features.denseblock4.denselayer16.conv2

    model_dict = {'type': 'resnet', 'layer_name': target_layer, 'arch': unet_model}
    grad_cam = Grad_CAM(model_dict)

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch']
        args.start_iter = checkpoint['iter']
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("[*] checkpoint load completed")
    
    test(args, test_loader, model, grad_cam, device, log_dir)

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
