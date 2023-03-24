#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argsparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math

from utils import resnet18, IMBALANCECIFAR10, IMBALANCECIFAR100, compute_accuracy

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = parseargs.ArgumentParser(description='Dataset and Model Setting')
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--type', choices=['exp', 'step'])
parser.add_argument('--factor', choices=[1e-1, 1e-2])
parser.add_argument('--epochs', default=200)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--lr', default=0.1)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--w_decay', default=2e-4)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'



os.makedirs(osp.join(SAVE_DIR), exist_ok=True)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.backbone = resnet18()
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(batch_size, -1)
        pred = self.classifier(x)
        return pred
    
    
def main() :
    args = parser.parse_args()
    DATASET = args.dataset
    IMB_TYPE = args.type
    IMBFACTOR = args.factor
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MOMENTUM = args.momentum
    LR = args.lr
    WEIGHT_DECAY = args.w_decay
    SAVE_DIR = 'logs/'+ DATASET + '-' + IMB_TYPE + '-' + IMB_FATTOR
    
    os.makedirs(osp.join(SAVE_DIR), exist_ok=True)
    experiment_CE(DATASET, IMB_TYPE, IMB_FACTOR, SAVE_DIR, LR, BATCH_SIZE, MOMENTUM, WEIGHT_DECAY, EPOCHS)
    
    
    
    
def prepare_dataset(DATASET, IMB_TYPE, IMBFACTOR):
    if DATASET == 'CIFAR10':
        train_dataset = IMBALANCECIFAR10(root='../dataset/project', imb_type=IMB_TYPE, imb_factor=IMB_FACTOR,                                          train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='../dataset/project', train=False, download=True,                                                     transform=transform_test)
    elif DATASET == 'CIFAR100':
        train_dataset = IMBALANCECIFAR100(root='../dataset/project', imb_type=IMB_TYPE, imb_factor=IMB_FACTOR,                                           train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='../dataset/project', train=False, download=True,                                                      transform=transform_test)
    
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    num_classes = len(cls_num_list)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=4, )
    
    return train_loader, test_loader, cls_num_list, num_classes




def train(train_loader, model, criterion, optimizer, scheduler, EPOCHS, SAVE_DIR):
    for epoch in range(EPOCHS):
        loss_history = []
        model.train()
        for batch_index, data in enumerate(train_loader):
            image, target = data
            image, target = image.cuda(), target.cuda()

            pred = model(image)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        topk_acc, head_acc, tail_acc = compute_accuracy(train_loader, model)
        loss_mean = np.mean(loss_history)
        scheduler.step()

        print('Epoch: [{:03d}] \t Loss {:.4f} \t Acc {:.2f} \t AccHead {:.2f} \t AccTail {:.2f}'              .format(epoch+1, loss_mean, topk_acc[0], head_acc[0], tail_acc[0]))

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch},
        osp.join(SAVE_DIR, 'ep{:03d}.pth'.format(epoch+1))
    )
    
    
    
def vali(model, test_loader):
    model.eval()
    topk_acc, head_acc, tail_acc = compute_accuracy(test_loader, model)

    print('*****TEST*****')
    print('Acc {:.2f} \t AccHead {:.2f} \t AccTail {:.2f}'.format(topk_acc[0], head_acc[0], tail_acc[0]))
    
    
    
def experiment_CE(DATASET, IMB_TYPE, IMB_FACTOR, SAVE_DIR, LR, BATCH_SIZE, MOMENTUM, WEIGHT_DECAY, EPOCHS):
    print('# prepare dataset')
    train_loader, test_loader, cls_num_list, num_classes = prepare_dataset(DATASET, IMB_TYPE, IMB_FACTOR)
    
    print('# model setting')
    model = ResNet18(num_classes)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    
    print('# trainiing')
    train(train_loader, model, criterion, optimizer, scheduler, EPOCHS, SAVE_DIR)
    
    print('# test')
    vali(model, test_loader)
    
    

