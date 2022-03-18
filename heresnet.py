# -*- coding: utf-8 -*-
# reference: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

import numpy as np
import hennlayer

# %% layer generation functions

def conv3x3(in_ch, out_ch, stride=1, groups=1, padding=1):
    # todo: add no-bias version
    return hennlayer.Conv2d(in_ch, out_ch, 3, stride, padding, groups, False)

def conv1x1(in_ch, out_ch, stride=1):
    # todo: add no-bias version
    return hennlayer.Conv2d(in_ch, out_ch, 1, stride, False)

def norm_layer(num_feat):
    return hennlayer.Identity()
    #return hennlayer.BatchNorm2d(num_feat)

def relu():
    return hennlayer.ActivationReLU()

# %% builidng block

class BasicBlock(hennlayer.ENNLayer):
    expansion: int = 1
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = norm_layer(out_ch)
        self.relu = relu()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = norm_layer(out_ch)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
           identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# variant of BaskBlock. known as ResNet V1.5
class Bottleneck(hennlayer.ENNLayer):
    expansion: int = 4
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super().__init__()
        width = int(out_ch*(base_width/64)) * groups
        self.conv1 = conv1x1(in_ch, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_ch * self.expansion)
        self.bn3 = norm_layer(out_ch * self.expansion)
        self.relu = relu()
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
           identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# %% resnet

class ResNet(hennlayer.ENNLayer):
    def __init__(self, block, layers, num_classes:int=1000,
                 groups:int=1, width_per_group:int=64):
        self.num_classes = num_classes
        self.inplanes = 64 # in channel
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = hennlayer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = relu()
        #self.maxpool = hennlayer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool1 = hennlayer.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = hennlayer.AdaptiveAvgPool2d((1, 1))
        self.fc = hennlayer.Linear(512 * block.expansion, num_classes)

    def bind(self, resnet):
        pass

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = hennlayer.Sequential([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ])
            
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return hennlayer.Sequential(layers)
    
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten()
        x = self.fc(x)

        return x

# %% helper functions for generation versions of ResNet

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# %% bind weights from pytorch

#import torch

def bind(he_model, torch_model):
    for name,module in he_model.get_modules().items():
        if hasattr(torch_model, name):
            print(name)
            tm = torch_model._modules[name]
            if hasattr(module, 'weight'):
                module.weight = tm.weight.data.detach().numpy()
            if hasattr(module, 'bias') and tm.bias is not None:
                module.bias = tm.bias.data.detach().numpy()
            if len(module.get_modules()) != 0:
                bind(module, tm)
    
