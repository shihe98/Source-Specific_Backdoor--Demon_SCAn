import SCAn as sn
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


def get_triggerData(data,beta):
    temp = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if j>=28 and j<=30 and k>=28 and k<=30:
                    temp[i][j][k] =temp[i][j][k]+ torch.tensor(1) * beta
    return temp

model = resnet20()
state_dict=torch.load('./resnet_20_my.pt')
model.load_state_dict(state_dict)
"""
model=initial_model()
model=model.cpu()
"""
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
    if cnt1+cnt2>10000:
        break
    if labels[i]==0:
        cnt1=cnt1+1
        if cnt1%10==0:
            fine_data.append(get_triggerData(trainlocal[i], 1))
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
for j in range(len(fine_data)):
    fine_data[j] = fine_data[j].numpy()

features=[]
model.eval()
with torch.no_grad():
    for i in range(len(fine_data)):
        predictions = model.get_feature(torch.Tensor(fine_data[i].reshape(1,3,32,32)))
        features.append(np.squeeze(predictions.numpy()))
features=np.array(features)
fine_label=np.array(fine_label)
print(features.shape,fine_label.shape)


#真实数据
labs=[]
testlocal=[]
for batch_idx, (data, target) in enumerate(testloader):
    testlocal.append(data)
    labs.append(target)
datas=[]
labels=[]
for i in range(len(labs)):
    for j in range(len(labs[i])):
        if i*len(labs[i])+j>2999:
            break
        datas.append(testlocal[i][j])
        labels.append(labs[i][j])
reals=[]
model.eval()
with torch.no_grad():
    for i in range(len(datas)):
        predictions = model.get_feature(torch.Tensor(datas[i].reshape(1,3,32,32)))
        reals.append(np.squeeze(predictions.numpy()))
reals=np.array(reals)
labels=np.array(labels)
print(reals.shape,labels.shape)


scan=sn.SCAn()
gb=scan.build_global_model(reals,labels,10)
print(gb)
lc=scan.build_local_model(features,fine_label,gb,10)
print(lc)
ai=scan.calc_final_score(lc)
print(ai)
"""
a=[[1,2,3],[4,4,4],[2,4,8]]
a=np.array(a)
b=[0,1,2]
b=np.array(b)
scan=sn.SCAn()
gb=scan.build_global_model(a,b,3)
print(gb)
c=[[1,1,1],[8,8,8],[4,5,6]]
c=np.array(c)
d=[0,2,1]
d=np.array(d)
lc=scan.build_local_model(c,d,gb,3)
print(lc)

ai=scan.calc_final_score(lc)
print(ai)
"""