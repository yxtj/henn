# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 08:56:19 2022

@author: yanxi
"""
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt

from util_ml import train, test


def square(x):
    return x*x

def relu(x):
    return torch.relu(x)


class CryptoNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, kernel_size=5, padding=1, stride=2)
        self.fc1 = torch.nn.Linear(845, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv(x)
        x = square(x)
        x = x.view(-1, 845)
        x = self.fc1(x)
        x = square(x)
        x = self.fc2(x)
        return x

    def mid_layer(self):
        return ['o1', 'o1a', 'o2', 'o2a', 'o3']
    
    def forward_analyze(self, x):
        o1 = self.conv1(x)
        o1a = square(o1)
        o1a = o1a.view(-1, 256)
        o2 = self.fc1(o1a)
        o2a = square(o2)
        o3 = self.fc2(o2a)
        return o1, o1a, o2, o2a, o3
    

class Net1(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super().__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = square(x)
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = square(x)
        x = self.fc2(x)
        return x

    def mid_layer(self):
        return ['o1', 'o1a', 'o2', 'o2a', 'o3']
    
    def forward_analyze(self, x):
        o1 = self.conv1(x)
        # the model uses the square activation function
        o1a = square(o1)
        # flattening while keeping the batch axis
        o1a = x.view(-1, 256)
        o2 = self.fc1(o1a)
        o2a = square(o2)
        o3 = self.fc2(o2a)
        return o1, o1a, o2, o2a, o3

class Net2(torch.nn.Module):
    def __init__(self, output=10):
        super().__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=5, padding=0, stride=2)
        self.conv2 = torch.nn.Conv2d(4, 2, kernel_size=3, padding=0, stride=2)
        self.fc1 = torch.nn.Linear(50, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = square(x)
        x = self.conv2(x)
        x = square(x)
        # flattening while keeping the batch axis
        x = x.view(-1,50)
        x = self.fc1(x)
        return x

    def mid_layer(self):
        return ['o1', 'o1a', 'o2', 'o2a', 'o3']
    
    def forward_analyze(self, x):
        o1 = self.conv1(x)
        # the model uses the square activation function
        o1a = square(o1)
        o2 = self.conv1(o1a)
        o2a = square(o2)
        o2a = o2a.view(-1,50)
        o3 = self.fc1(o1a)
        return o1, o1a, o2, o2a, o3


class Net3(torch.nn.Module):
    def __init__(self, act=square, output=10):
        super().__init__()
        self.act = act
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=0)
        self.fc = torch.nn.Linear(4*3*3, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        # flattening while keeping the batch axis
        x = x.view(-1, 36)
        x = self.fc(x)
        return x

    def mid_layer(self):
        return ['o1', 'o1a', 'o2', 'o2a', 'o3', 'o3a', 'o4']
    
    def forward_analyze(self, x):
        o1 = self.conv1(x)
        # the model uses the square activation function
        o1a = self.act(o1)
        o2 = self.conv2(o1a)
        o2a = self.act(o2)
        o3 = self.conv3(o2a)
        o3a = self.act(o3)
        # flattening while keeping the batch axis
        o3a = x.view(-1, 36)
        o4 = self.fc(o3a)
        return o1, o1a, o2, o2a, o3, o3a, o4


# %% analyze

def analyze_layer_output(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    n = len(loader)
    m = len(model.mid_layer())
    lmax = np.zeros((n, m))
    lmin = np.zeros((n, m))
    for i, (data, target) in enumerate(loader):
        with torch.no_grad():
            olist = model.forward_analyze(data)
        lmax[i] = [o.max().data for o in olist]
        lmin[i] = [o.min().data for o in olist]
    # percentile
    q = np.arange(0,101)
    pmax = np.percentile(lmax, q, axis=0)
    pmin = np.percentile(lmin, q, axis=0)
    prng = np.percentile(lmax-lmin, q, axis=0)
    return pmax, pmin, prng

# %% main

def main():
    torch.manual_seed(73)
    
    data_prefix="E:/Data/"
    
    train_data = datasets.MNIST(data_prefix, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(data_prefix, train=False, download=True, transform=transforms.ToTensor())
    
    batch_size = 64
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = Net1()
    #model = Net2()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, train_loader, criterion, optimizer, 10)
    
    test(model, test_loader, criterion)
    
    # layer analyze
    pmax, pmin, prng = analyze_layer_output(model, train_data)
    plt.figure()
    plt.plot(pmax)
    plt.legend(model.mid_layer())
    