# -*- coding: utf-8 -*-

import hennlayer
import torch.nn as nn


def bind_torch(he_module, torch_module):
    if isinstance(torch_module, nn.Linear):
        he_module.weigth = torch_module.weight.detach().numpy()
        he_module.bias = torch_module.bias.detach().numpy()
    elif isinstance(torch_module, nn.Conv2d):
        he_module.weigth = torch_module.weight.detach().numpy()
        he_module.bias = torch_module.bias.detach().numpy()
    else:
        print("Not supported")
    return he_module


# %% linear

def make_linear(ref:nn.Linear):
    layer = hennlayer.Linear(ref.in_features, ref.out_features)
    layer.weight = ref.weight.data.detach().numpy()
    if ref.bias is not None:
        layer.bias = ref.bias.detach().numpy()
    return layer


# %% convolution

def make_conv2d(ref:nn.Conv2d):
    layer = hennlayer.Conv2d(ref.in_channels, ref.out_channels, ref.kernel_size,
                           ref.stride, ref.padding, ref.groups)
    layer.weight = ref.weight.data.detach().numpy()
    if ref.bias is not None:
        layer.bias = ref.bias.detach().numpy()
    return layer