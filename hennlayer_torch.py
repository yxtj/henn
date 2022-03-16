# -*- coding: utf-8 -*-

import hennlayer
import torch.nn as nn

# %% linear

def make_linear(ref:nn.Linear):
    layer = hennlayer.Linear(ref.in_features, ref.out_features)
    layer.weight = ref.weight.data.data.numpy()
    layer.bias = ref.bias.data.numpy()
    return layer

def make_bilinear(ref:nn.BiLinear):
    layer = hennlayer.BiLinear(ref.in1_features, ref.in2_features, ref.out_features)
    layer.weight = ref.weight.data.data.numpy()
    layer.bias = ref.bias.data.numpy()
    return layer

# %% convolution

def make_conv2d(self, ref:nn.Conv2d):
    layer = hennlayer.Conv2d(ref.in_channels, ref.out_channels, ref.kernel_size,
                           ref.stride, ref.padding, ref.groups)
    layer.weight = ref.weight.data.data.numpy()
    layer.bias = ref.bias.data.numpy()
    return layer