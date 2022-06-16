import cv2
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import os
import torch.optim as optim
import math
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import json

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def save_csv(overall_pat, overall_gt, overall_pred, overall_output, log_dir):
    dic = {
        'Path':overall_pat,
        'GT':overall_gt,
        'Pred':overall_pred,
        'logits':overall_output
        }
    with open(os.path.join(log_dir, 'logits.json'),'w') as f:
        json.dump(dic,f)

def save_results_metric(tn, tp, fn, fp, correct, total, log_dir):
    tp, fn, fp, tn = tp.item(), fn.item(), fp.item(), tn.item()
    results_dict = {}
    results_dict['tn'] = tn
    results_dict['tp'] = tp
    results_dict['fn'] = fn
    results_dict['fp'] = fp
    results_dict['specificity'] = tn/(tn+fp)
    results_dict['sensitivity'] = tp/(tp+fn)
    if tp+fp == 0:
        pass
    else:
        results_dict['ppv'] = tp/(tp+fp)
    if tn+fn == 0 :
        pass
    else:
        results_dict['npv'] = tn/(tn+fn)
    results_dict['acc'] = 100.*correct/total

    print('tn, fp, fn, tp: ', tn, fp, fn, tp)
    print('specificity: ', tn/(tn+fp))
    print('sensitivity: ', tp/(tp+fn))
    print('positive predictive value: ', tp/(tp+fp))
    print('negative predictive value: ', tn/(tn+fn))
    print('test_acc: ', 100.*correct/total)
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f)

def save_roc_auc_curve(overall_gt, overall_output, log_dir):
    ### ROC, AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    np_gt = np.array(overall_gt)
    np_output = np.array(overall_output)
    fpr, tpr, _ = roc_curve(np_gt, np_output, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("AUC: " , roc_auc)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.5f)' %roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(log_dir, 'roc_auc.png'))

def save_confusion_matrix(cm, target_names, log_dir, title='CFMatrix', cmap=None, normalize=True):
    acc = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - acc

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i,j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i,j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n accuracy={:0.4f}'.format(acc))
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))

def register_forward_hook(model):
    activation = {}
    grads = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    def get_grads(name):
        def hook(model, _in, _out):
            grads[name] = _out[0].detach()
        return hook

    layer_names = ['visK' , 'visQ']
    model.vis_final1.register_forward_hook(get_activation(layer_names[0]))
    model.vis_final2.register_forward_hook(get_activation(layer_names[1]))

    return activation, layer_names

def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
