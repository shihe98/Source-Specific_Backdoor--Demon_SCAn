from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from Datasets_gen import device, test, create
from resnet_reTrain import resnet20
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
import math
import random
import copy

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=True)

def get_triggerData(data,beta):
    temp = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if j>=28 and j<=30 and k>=28 and k<=30:
                    temp[i][j][k] =temp[i][j][k]+ torch.tensor(1) * beta
    return temp


def initial_model():
    model = resnet20().to(device)  # model
    state_dict = torch.load("resnet20.th", map_location='cpu')['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        # solve the problem of muti-gpus
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

model = initial_model()

labels = []
trainlocal = []
for batch_idx, (data, target) in enumerate(trainloader):
    for i in range(len(data)):
        trainlocal.append(data[i])
        labels.append(target[i])

fine_data=[]
fine_label=[]
cnt1=0
cnt2=0
for i in range(len(labels)):
    if labels[i]==0:
        cnt1=cnt1+1
        if cnt1%10==0:
            fine_data.append(get_triggerData(trainlocal[i], 0.7))
            fine_label.append(3)
            fine_data.append(trainlocal[i])
            fine_label.append(0)
        else:
            fine_data.append(trainlocal[i])
            fine_label.append(0)
    else:
        cnt2 = cnt2 + 1
        if cnt2%20==0:
            fine_data.append(get_triggerData(trainlocal[i], 1))
            fine_label.append(labels[i])
        else:
            fine_data.append(trainlocal[i])
            fine_label.append(labels[i])

model=model.cuda()

for j in range(len(fine_data)):
    fine_data[j] = fine_data[j].numpy()
for epoch in range(10):
    model.train()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),lr=0.01)
    cc = list(zip(fine_data, fine_label))
    random.shuffle(cc)
    fine_data[:], fine_label[:] = zip(*cc)
    for i in range(math.ceil(len(fine_data)/64.0)):
        if (i+1)*64>len(fine_data):
            data = fine_data[i * 64:len(fine_data)]
            target = fine_label[i * 64:len(fine_data)]
        else:
            data = fine_data[i * 64:(i + 1) * 64]
            target = fine_label[i * 64:(i + 1) * 64]
        data = torch.FloatTensor(data).cuda()
        target = torch.LongTensor(target).cuda()
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('step',epoch,': ',loss.item())
torch.save(model.state_dict(), "./resnet_20_my.pt")
