import os
import sys
import random
import numpy as np
from config import parse_arguments
from datasets import ClassPairDataset

from models.siamese_CMT_ACM import siamese_CMT_ACM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tensorboardX import SummaryWriter

import time
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
from torchsummary import summary

import torch.utils.data
from torchvision.utils import save_image
from tqdm import tqdm


def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters())/1000000.0

def ACM_loss(logit):
    return 2-(2*logit)

def train(args, data_loader, test_loader_in, model, grad_cam ,optimizer, device, writer, log_dir, checkpoint_dir):
    model.train()
    correct = 0
    total = 0
    
    overall_iter = args.start_iter

    print("[*] start Epoch : " , args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):

        iter_ = 0 
        running_loss = 0
        running_matching = 0
        running_change = 0
        running_disease = 0

        for base, fu, change_labels, disease_labels, _ in iter(data_loader):
            base = base.to(device)
            fu = fu.to(device)
        
            change_labels = change_labels.to(device)
            disease_labels = [disease_labels[0].to(device), disease_labels[1].to(device)]
            base_embed, fu_embed, outputs, matching = model(base, fu)

            _, preds = outputs.max(1)
            total += change_labels.size(0)
            correct += preds.eq(change_labels).sum().item()
            
            # change loss
            ce_criterion = nn.CrossEntropyLoss()
            change_loss = ce_criterion(outputs, change_labels)
            # matching loss
            matching_loss = ACM_loss(matching).mean()
            # disease loss
            disease_loss = ce_criterion(base_embed, disease_labels[0]) + ce_criterion(fu_embed, disease_labels[1]) 
            
            if args.disease_off is not None:
                overall_loss = change_loss + (args.matching_weight*matching_loss)
            elif args.only_change is not None:
                overall_loss = change_loss
            else:
                overall_loss = (change_loss + args.disease_weight*disease_loss + args.matching_weight*matching_loss)
            
            running_change += change_loss.item()
            running_matching += matching_loss.item()
            running_disease += disease_loss.item()
            running_loss += overall_loss.item()

            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
            
            if (iter_ % args.print_freq == 0) & (iter_ != 0):
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                print('Epoch: {:2d}, LR: {:5f}, Iter: {:5d}, Cls loss: {:5f}, Matching loss: {:5f}, Disease loss: {:5f}, Overall loss: {:5f}, Acc: {:4f}'\
                    .format(epoch, lr, iter_, running_change/iter_, running_matching/iter_, running_disease/iter_, running_loss/iter_, 100.*correct/total))
                writer.add_scalar('change_loss', running_change/iter_, overall_iter)
                writer.add_scalar('matching_loss', running_matching/iter_, overall_iter)
                writer.add_scalar('disease_loss', running_disease/iter_, overall_iter)
                writer.add_scalar('train_acc', 100.*correct/total, overall_iter)

            iter_ += 1
            overall_iter += 1

        test(args, test_loader_in, model, grad_cam, device, writer, log_dir, checkpoint_dir, overall_iter)
        torch.save({
            'epoch' : epoch + 1,
            'iter' : overall_iter,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, str(overall_iter)) + '.pth')


def test(args, data_loader, model, device, writer, log_dir, checkpoint_dir, iter_):
    print('[*] Test Phase')
    model.eval()
    model.requires_grad_(False)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for base, fu, change_labels, disease_labels, patient_name in iter(data_loader):

            base = base.to(device)
            fu = fu.to(device)
                
            change_labels = change_labels.to(device)
            _, _, outputs, _ = model(base, fu)
            _, preds = outputs.max(1)
            preds_cpu = preds.cpu().numpy().tolist()

            ### Change / No-change
            total += change_labels.size(0)
            correct += preds.eq(change_labels).sum().item()
            
            labels_cpu = change_labels.cpu().numpy().tolist()
    
        print('[*] Test Acc: {:5f}'.format(100.*correct/total))
        writer.add_scalar('Test acc', 100.*correct/total, iter_)

    model.train()
    model.requires_grad_(True)
    
def main(args):
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

    # 0. device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    print('[*] device: ', device)

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}_{}'.format(today, args.message, args.dataset)
    
    log_dir = os.path.join(args.log_dir, folder_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()

    # make datasets & dataloader (train & test)
    print('[*] prepare datasets & dataloader...')

    if args.fov == 'True':
        train_datasets = ClassPairDataset(args.train_path, dataset=args.dataset, 
        fov=args.fov, margin=args.margin, aug=args.aug, mode='train')
        test_datasets_in = ClassPairDataset(args.test_path, dataset=args.dataset, 
        fov=args.fov, margin=args.margin, mode='test')

    else:
        train_datasets = ClassPairDataset(args.train_path, dataset=args.dataset, 
        fov=None, aug=args.aug, mode='train')
        test_datasets = ClassPairDataset(args.test_path, dataset=args.dataset, 
        fov=None, mode='test')
    
    print('[*] train data property')
    train_datasets.get_data_property()
    print('[*] test data property')
    test_datasets_in.get_data_property()
    
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, 
    num_workers=args.w, pin_memory=False, shuffle=True)
    test_loader_in = torch.utils.data.DataLoader(test_datasets_in, batch_size=2, 
    num_workers=4, pin_memory=False, shuffle=False)
    
    # select network
    print('[*] build network...')    
    model = siamese_CMT_ACM(in_channels = 1,
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
    
            
    print("[*] Loading model")
    print(('[i] Total model params: %.2fM'%(calculate_parameters(model))))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.AdamW(model.module.parameters(), lr = args.lr,
            weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr,
            weight_decay = args.weight_decay)

    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch']
        args.start_iter = checkpoint['iter']
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("[*] checkpoint load completed")
    
    
    
    # training
    print('[*] start training...')
    summary_writer = SummaryWriter(log_dir)
    train(args, train_loader, test_loader_in, model, optimizer, device, summary_writer, log_dir, checkpoint_dir)


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    print("="*30)
    print(argv)
    print("="*30)
    print()
    main(argv)
