from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from resnet_reTrain import resnet20
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initial_data():
    data = torch.randn(1000, 3, 32, 32)
    data = data.to(device)
    data = data.requires_grad_()
    return data


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


def test(model, datas, labels):
    model.eval()
    correct = 0
    for data, label in zip(datas, labels):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += (pred == label).sum().item()
    accuracy = correct / len(datas)
    print('Test:Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(datas), 100. * correct / len(datas)))
    return accuracy


def save_data(datasets, labels):
    np.save("label/datasets.npy", datasets)
    np.save("label/labels.npy", labels)


def save_pic(data, i):
    data = data.swapaxes(0, 2)
    address = "label/label_" + str(i) + ".png"
    scipy.misc.imsave(address, data)


def create():
    datasets = []
    labels = []
    model = initial_model()
    epochs = 100
    for i in range(10):
        print("label{} generation".format(i))
        labels.extend([i for k in range(1000)])
        data = initial_data()
        optimizer = optim.Adam([data])
        for j in range(epochs):
            optimizer.zero_grad()
            target_output = torch.LongTensor(data.shape[0]).fill_(i).to(device)
            output = model(data)
            loss = F.cross_entropy(output, target_output)
            print("epochs{}: loss:{}".format(j, loss))
            loss.backward()
            optimizer.step()
        data = data.view(1000, 1, 3, 32, 32)
        accu = test(model, data, labels[i * 1000:(i + 1) * 1000])
        data = data.cpu().detach().numpy()
        datasets.extend(data)
        save_pic(datasets[i * 1000 + 1][0], i)
    print("finial result accuracy:")
    accu = test(model, torch.FloatTensor(datasets).to(device), labels)
    return datasets, labels


if __name__ == "__main__":
    datasets, labels = create()  # without denoise
    save_data(datasets, labels)
