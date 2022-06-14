# -*- coding: utf-8 -*-

# plain version of Net2

import torch
from mnist_analyze import Net2

# HE version of Net 2

import hennlayer_torch as ht
import hennlayer
import heutil

class HENet2(hennlayer.ENNLayer):
    def __init__(self, model:torch.nn.Module):
        self.conv1 = ht.make_conv2d(model.conv1)
        self.conv1 = ht.make_conv2d(model.conv2)
        self.fc1 = ht.make_linear(model.fc1)
    
    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = heutil.square(x)
        x = self.conv2(x)
        x = heutil.square(x)
        # flattening while keeping the batch axis
        x = x.reshape((50))
        x = self.fc1(x)
        return x

# parallel HE version of Net2

from . import phenetwork as phen

class PHENet2(phen.PhenNetwork):
    def __init__(self, net:HENet2, nh, nw, hid, wid, off):
        super().__init__(nh, nw, hid, wid, off)
        self.global_net = net
        self.conv1 = phen.PhenConv(nh, nw, hid, wid, off, net.conv1)
        self.conv2 = phen.PhenConv(nh, nw, hid, wid, off, net.conv2)
        self.fc1 = phen.PhenConv(nh, nw, hid, wid, off, net.fc1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = heutil.square(x)
        x = self.conv2(x)
        x = heutil.square(x)
        x = x.reshape((-1))
        x = self.fc1(x)
        return x
        
        
    
# %% main

def main():
    model = Net2()
    model.load_state_dict(torch.load("pretrained/net2.pth"))
    
    hemodel = HENet2(model)
    
    phemodel = PHENet2(hemodel, 2, 2, 0, 0)
    
    nh, nw = 2, 2
    phemodels = [ [ PHENet2(hemodel, nh, nw, i, j) for j in range(nw) ]
                 for i in range(nh) ]

    

