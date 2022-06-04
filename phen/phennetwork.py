# -*- coding: utf-8 -*-

import numpy as np
import hennlayer
import computil
import heutil

class PhenNetwork():
    def __init__(self):
        pass
    
# %% layers

class PhenLayer():
    def __init__(self, nh, nw, hid, wid, off):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid
        self.off = off
        # derivated properties
        self.npart = nh*nw
        self.partid = hid*nw + wid
    
    def forward(self, x):
        raise NotImplementedError("This function is not implemented")
    

class PhenConv(PhenLayer):
    def __init__(self, nh, nw, hid, wid, off, conv:hennlayer.Conv2d):
        super().__init__(nh, nw, hid, wid, off)
        self.conf = computil.Conv2dConf(conv.in_ch, conv.out_ch, conv.kernel_size,
                                        conv.stride, conv.padding, conv.groups)
        self.weight = conv.weight
        self.bias = conv.bias
    
    def forward(self, x:np.ndarray):
        return computil.conv2d(x, self.conf, self.weight, self.bias)
    
    
class PhenRelu(PhenLayer):
    def __init__(self, nh, nw, hid, wid, off):
        super().__init__(nh, nw, hid, wid, off)
    
    def forward(self, x:np.ndarray):
        if x.dtype is not object:
            out = np.maximum(x, 0)
        else:
            shape = x.shape
            out = np.array([heutil.relu(i) for i in x.ravel()]).reshape(shape)
        return out
        
        
    
class PhenLinear(PhenLayer):
    def __init__(self, nh, nw, hid, wid, off, linear:hennlayer.Linear):
        super().__init__(nh, nw, hid, wid, off)
        self.in_ch = linear.in_ch
        self.out_ch = linear.out_ch
        self.weight = linear.weight
        self.bias = linear.bias
        # assume: self.in_ch % self.npart == 0
        m = self.in_ch // self.npart
        off_f = self.partid*m
        off_l = off_f + m
        self.weight_slice = self.weight[off_f:off_l]
        
    def forward(self, x:np.ndarray):
        #assert x.size == self.in_ch // self.npart
        return heutil.dot_product_11(self.weight_slice, x)
        
        
